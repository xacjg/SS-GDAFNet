import numpy
import numpy as np
import torch
import random
import cv2


class Scale(object):
    def __init__(self, wi, he):
        self.w = wi
        self.h = he

    def __call__(self, img, label):
        img = cv2.resize(img, (self.w, self.h))

        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [img, label]


class Resize(object):
    def __init__(self, min_size, max_size, strict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.strict = strict

    def get_size(self, image_size):
        w, h = image_size
        if not self.strict:
            size = random.choice(self.min_size)
            max_size = self.max_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)
        else:
            if w < h:
                return (self.max_size, self.min_size[0])
            else:
                return (self.min_size[0], self.max_size)

    def __call__(self, image, label):
        size = self.get_size(image.shape[:2])
        image = cv2.resize(image, size)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return (image, label)

class RandomCropResize(object):
    def __init__(self, crop_area):
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h - y1, x1:w - x1]
            label_crop = label[y1:h - y1, x1:w - x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)

            return img_crop, label_crop
        else:
            return [img, label]


class RandomFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
                image = cv2.flip(image, 0)  # horizontal flip
                label = cv2.flip(label, 0)  # horizontal flip
        if random.random() < 0.5:
                image = cv2.flip(image, 1)  # veritcal flip
                label = cv2.flip(label, 1)  # veritcal flip
        return [image, label]


class RandomExchange(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            pre_img = image[:, :, 0:3]
            post_img = image[:, :, 3:6]
            image = numpy.concatenate((post_img, pre_img), axis=2)
        return [image, label]


class Normalize(object):
    def __init__(self, mean, std):

        self.mean = mean
        self.std = std
        self.depth_mean = [0.5]
        self.depth_std = [0.5]

    def __call__(self, image, label):
        image = image.astype(np.float32)
        image = image / 255
        label = np.ceil(label / 255)
        for i in range(6):
            image[:, :, i] -= self.mean[i]
        for i in range(6):
            image[:, :, i] /= self.std[i]

        return [image, label]


class GaussianNoise(object):
    def __init__(self, std=0.05):

        self.std = std

    def __call__(self, image, label):
        noise = np.random.normal(loc=0, scale=self.std, size=image.shape)
        image = image + noise.astype(np.float32)
        return [image, label]


class ToTensor(object):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w / self.scale), int(h / self.scale)), \
                               interpolation=cv2.INTER_NEAREST)
        image = image[:, :, ::-1].copy()  # .copy() is to solve "torch does not support negative index"
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)

        label_tensor = torch.LongTensor(np.array(label, dtype=np.int)).unsqueeze(dim=0)

        return [image_tensor, label_tensor]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
