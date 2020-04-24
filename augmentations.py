import torchvision.transforms.functional as TF
import numpy as np
import torch


class Numpy2Torch(object):
    def __call__(self, sample):
        img, label = sample
        img_tensor = TF.to_tensor(img)
        label_tensor = TF.to_tensor(label)
        return img_tensor, label_tensor


# Performs uniform cropping on images
class UniformCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, img: torch.Tensor, label: torch.Tensor):
        _, height, width = img.shape
        crop_limit_x = width - self.crop_size
        crop_limit_y = height - self.crop_size
        x = np.random.randint(0, crop_limit_x)
        y = np.random.randint(0, crop_limit_y)

        img = img[:, y:y+self.crop_size, x:x+self.crop_size]
        label = label[:, y:y+self.crop_size, x:x+self.crop_size]
        return img, label

    def __call__(self, sample):
        img, label = sample
        img, label = self.random_crop(img, label)
        return img, label


class ImportanceRandomCrop(UniformCrop):
    def __call__(self, sample):

        img, label = sample

        sample_size = 5
        balancing_factor = 200

        random_crops = [self.random_crop(img, label) for i in range(sample_size)]
        crop_weights = np.array([crop_label.sum() for crop_img, crop_label in random_crops]) + balancing_factor
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(sample_size, p=crop_weights)
        img, label = random_crops[sample_idx]

        return img, label


class RandomFlip(object):
    def __call__(self, sample):
        img, label = sample
        h_flip = np.random.choice([True, False])
        v_flip = np.random.choice([True, False])

        if h_flip:
            img = np.flip(img, axis=0)
            label = np.flip(label, axis=0)

        if v_flip:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)

        return img, label


class RandomRotate(object):
    def __call__(self, sample):
        img, label = sample
        rotation = np.random.randint(0, 360)
        img = ndimage.rotate(img, rotation, reshape=False).copy()
        label = ndimage.rotate(label, rotation, reshape=False).copy()
        return img, label
