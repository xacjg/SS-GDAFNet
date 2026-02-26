import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet18
from models.decoder1 import FlowchartNet256




class ECAChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ECAChannelAttentionModule, self).__init__()

        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                              groups=in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        batch_size, channels, height, width = x.size()
        x_reshaped = x.view(batch_size, channels, -1)
        x_conv = self.conv(x_reshaped)
        attention = self.sigmoid(x_conv)

        attention = attention.view(batch_size, channels, height, width)
        return x * attention


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.maxpool(x)  # 输出维度： (batch_size, in_channels, 1, 1)
        avg_out = self.avgpool(x)  # 输出维度： (batch_size, in_channels, 1, 1)
        out = torch.cat([max_out, avg_out], dim=1)  # 在通道维度拼接
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out


class EnhancedAttentionModule(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(EnhancedAttentionModule, self).__init__()
        self.channel_attention = ECAChannelAttentionModule(in_channels, kernel_size)
        self.spatial_attention = SpatialAttentionModule(in_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        combined = channel_out + spatial_out
        output = self.conv(combined)
        return output

class DML(nn.Module):
    def __init__(self,in_channels):
        super(DML, self).__init__()

        self.adaptm = nn.AdaptiveMaxPool2d(1)

        self.upper_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.lower_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.dim_reduction_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.reduce = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.attention_module = EnhancedAttentionModule(in_channels = in_channels, kernel_size=3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):

        feat1_pool = self.adaptm(x1)
        feat2_pool = self.adaptm(x2)

        concat_features = torch.cat([x1, x2], dim=1)
        concat_features_3 = self.conv3(concat_features)
        concat_features_1 = self.conv1(concat_features)
        concat_features = concat_features_3 + concat_features_1
        gated_features = self.attention_module(concat_features)

        processed = gated_features
        processed_feat1 = processed * feat1_pool
        processed_feat2 = processed * feat2_pool
        processed_feat1 = processed_feat1 + processed
        processed_feat2 = processed_feat2 + processed
        out = processed_feat1 - processed_feat2
        change_map = self.output_conv(out)
        return change_map

class BaseNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(BaseNet, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.mid_d = 64
        self.decoder = Decoder(self.mid_d, output_nc)
        self.DML2 = DML(64)
        self.DML3 = DML(128)
        self.DML4 = DML(256)
        self.DML5 = DML(512)
        self.channel_reducer_512to64 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.channel_reducer_256to64 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.channel_reducer_128to64 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.FlowchartNet256 = FlowchartNet256()

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone.base_forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone.base_forward(x2)

        d5_1 = self.DML5(x1_5, x2_5)
        d4_1 = self.DML4(x2_4, x1_4)
        d3_1 = self.DML3(x2_3, x1_3)
        d2 = self.DML2(x2_2, x1_2)

        fusion_features = [d2, d3_1, d4_1, d5_1]
        mask = self.FlowchartNet256(fusion_features)

        return mask