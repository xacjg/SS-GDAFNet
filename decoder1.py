import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=nn.ReLU(inplace=True)):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 512), fpn_channels=64):
        super().__init__()
        c3, c4, c5, c6 = in_channels

        self.lat6 = nn.Conv2d(c6, c5, 1, bias=False)  # 512->256
        self.smooth5 = ConvBNAct(c5, c5, 3)  # 平滑256通道

        self.lat5 = nn.Conv2d(c5, c4, 1, bias=False)  # 256->128
        self.smooth4 = ConvBNAct(c4, c4, 3)  # 平滑128通道

        self.lat4 = nn.Conv2d(c4, c3, 1, bias=False)  # 128->64
        self.smooth3 = ConvBNAct(c3, c3, 3)  # 平滑64通道

        self.final_proj = None
        if c3 != fpn_channels:
            self.final_proj = nn.Conv2d(c3, fpn_channels, 1, bias=False)

    def forward(self, feats):
        if isinstance(feats, (list, tuple)):
            c3, c4, c5, c6 = feats
        else:
            c3, c4, c5, c6 = feats["c3"], feats["c4"], feats["c5"], feats["c6"]

        p6_proj = self.lat6(c6)  # 512->256, (N,256,8,8)
        p6_up = F.interpolate(p6_proj, scale_factor=2, mode="bilinear", align_corners=False)
        p5 = c5 + p6_up
        p5 = self.smooth5(p5)  # 平滑, (N,256,16,16)

        p5_proj = self.lat5(p5)  # 256->128, (N,128,16,16)
        p5_up = F.interpolate(p5_proj, scale_factor=2, mode="bilinear", align_corners=False)
        p4 = c4 + p5_up  # 加法融合
        p4 = self.smooth4(p4)  # 平滑, (N,128,32,32)

        p4_proj = self.lat4(p4)  # 128->64, (N,64,32,32)
        p4_up = F.interpolate(p4_proj, scale_factor=2, mode="bilinear", align_corners=False)
        p3 = c3 + p4_up  # 加法融合
        p3 = self.smooth3(p3)  # 平滑, (N,64,64,64)

        if self.final_proj is not None:
            p3 = self.final_proj(p3)

        return p3  # (N,fpn_channels,64,64)


class MultiKernelHead(nn.Module):
    def __init__(self, in_ch=64, mid_ch=64, out_ch=64):
        super().__init__()
        self.branch3 = ConvBNAct(in_ch, mid_ch, k=3)
        self.branch5 = ConvBNAct(in_ch, mid_ch, k=5)
        self.branch7 = ConvBNAct(in_ch, mid_ch, k=7)
        self.fuse = nn.Conv2d(mid_ch * 3, out_ch, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        x = torch.cat([b3, b5, b7], dim=1)  # (N,3*mid,64,64)
        logits_64 = self.fuse(x)  # (N,out_ch,64,64) —— logits
        return logits_64


class UpToMask(nn.Module):
    def __init__(self, mid_ch=64, num_classes=1):
        super().__init__()
        self.smooth128 = ConvBNAct(mid_ch, mid_ch, k=3)
        self.smooth256 = ConvBNAct(mid_ch, mid_ch, k=3)
        self.proj = nn.Conv2d(mid_ch, num_classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.act_out = nn.Sigmoid()

    def forward(self, logits_64):
        x = F.interpolate(logits_64, scale_factor=2, mode="bilinear", align_corners=False)  # 128x128
        x = self.smooth128(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 256x256
        x = self.smooth256(x)
        x = self.proj(x)  # (N,num_classes,256,256) —— logits
        prob = self.act_out(x)  # **最终 Sigmoid** 概率 mask
        return prob, x  # (prob, logits)


class FrequencyBranch(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.high_filter = ConvBNAct(in_ch, out_ch // 2, k=1)  # 高频（边缘）
        self.low_filter = ConvBNAct(in_ch, out_ch // 2, k=1)  # 低频（结构）

    def forward(self, x):
        # 快速傅里叶变换
        fft = torch.fft.rfft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

        # 分离高低频
        h, w = fft_shift.shape[-2:]
        mask_low = torch.zeros_like(fft_shift)
        mask_low[..., h // 4:3 * h // 4, w // 4:3 * w // 4] = 1

        low_freq = torch.fft.ifftshift(fft_shift * mask_low, dim=(-2, -1))
        high_freq = torch.fft.ifftshift(fft_shift * (1 - mask_low), dim=(-2, -1))

        low = torch.fft.irfft2(low_freq, s=x.shape[-2:], norm='ortho')
        high = torch.fft.irfft2(high_freq, s=x.shape[-2:], norm='ortho')

        return torch.cat([self.low_filter(low), self.high_filter(high)], dim=1)
    
class LightFrequencyBranch(nn.Module):
    def __init__(self, in_ch=64, out_ch=64):
        super().__init__()
        self.high_branch = nn.Sequential(
            ConvBNAct(in_ch, out_ch // 2, k=3),
            ConvBNAct(out_ch // 2, out_ch // 2, k=3)  # 增加深度
        )
        self.low_branch = nn.Sequential(
            ConvBNAct(in_ch, out_ch // 2, k=5),  # 低频用大核
            ConvBNAct(out_ch // 2, out_ch // 2, k=3)
        )

        self.interaction = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),  # DW卷积
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False)  # PW卷积
        )
        
    def forward(self, x):
        fft = torch.fft.rfft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        
        h, w = fft_shift.shape[-2:]
        mask_low = torch.zeros_like(fft_shift)
        mask_low[..., h//4:3*h//4, w//4:3*w//4] = 1
        
        low = torch.fft.irfft2(
            torch.fft.ifftshift(fft_shift * mask_low, dim=(-2, -1)),
            s=x.shape[-2:], norm='ortho'
        )
        high = torch.fft.irfft2(
            torch.fft.ifftshift(fft_shift * (1 - mask_low), dim=(-2, -1)),
            s=x.shape[-2:], norm='ortho'
        )

        high_feat = self.high_branch(high)
        low_feat = self.low_branch(low)

        freq_feat = torch.cat([high_feat, low_feat], dim=1)
        out = self.interaction(freq_feat)
        
        return out



class FlowchartNet256(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 512), fpn_channels=64,
                 head_mid=64, num_classes=1):
        super().__init__()
        self.fpn = FPN(in_channels, fpn_channels)

        self.spatial_head = MultiKernelHead(fpn_channels, head_mid, head_mid)

        self.freq_branch = LightFrequencyBranch(fpn_channels, head_mid)

        self.fusion = nn.Sequential(
            nn.Conv2d(head_mid * 2, head_mid, 1, bias=False),
            nn.BatchNorm2d(head_mid),
            nn.ReLU(inplace=True)
        )

        self.up = UpToMask(mid_ch=head_mid, num_classes=num_classes)

    def forward(self, feats, return_logits=False):
        p3 = self.fpn(feats)  # (N,64,64,64)

        spatial_feat = self.spatial_head(p3)  # 空域多尺度
        freq_feat = self.freq_branch(p3)  # 频域特征

        fused = torch.cat([spatial_feat, freq_feat], dim=1)  # cat融合
        logits_64 = self.fusion(fused)  # 降维到64通道

        prob, logits_256 = self.up(logits_64)
        return (prob, logits_256) if return_logits else prob