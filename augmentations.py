import torchvision.transforms.functional as TF
import numpy as np


class Numpy2Torch(object):
    def __call__(self, args):
        input, label, image_path = args
        input_t = TF.to_tensor(input)
        label = TF.to_tensor(label)
        return input_t, label, image_path


# Performs uniform cropping on images
class UniformCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, input, label):
        image_size = input.shape[-2]
        crop_limit = image_size - self.crop_size
        x, y = np.random.randint(0, crop_limit, size=2)

        input = input[y:y+self.crop_size, x:x+self.crop_size, :]
        label = label[y:y+self.crop_size, x:x+self.crop_size]
        return input, label

    def __call__(self, args):
        input, label, image_path = args
        input, label = self.random_crop(input, label)
        return input, label, image_path


class ImportanceRandomCrop(UniformCrop):
    def __call__(self, args):
        input, label, image_path = args

        SAMPLE_SIZE = 5  # an arbitrary number that I came up with
        BALANCING_FACTOR = 200

        random_crops = [self.random_crop(input, label) for i in range(SAMPLE_SIZE)]
        # TODO Multi class vs edge mask
        weights = []
        for input, label in random_crops:
            if label.shape[2] >= 4:
                # Damage detection, multi class, excluding backround
                weights.append(label[...,:-1].sum())
            elif label.shape[2] > 1:
                # Edge Mask, excluding edge masks
                weights.append(label[...,0].sum())
            else:
                weights.append(label.sum())
        crop_weights = np.array([label.sum() for input, label in random_crops]) + BALANCING_FACTOR
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(SAMPLE_SIZE, p=crop_weights)
        input, label = random_crops[sample_idx]

        return input, label, image_path


class RandomFlipRotate(object):
    def __call__(self, args):
        input, label, image_path = args
        _hflip = np.random.choice([True, False])
        _vflip = np.random.choice([True, False])
        _rot = np.random.randint(0, 360)

        if _hflip:
            input = np.flip(input, axis=0)
            label = np.flip(label, axis=0)

        if _vflip:
            input = np.flip(input, axis=1)
            label = np.flip(label, axis=1)

        input = ndimage.rotate(input, _rot, reshape=False).copy()
        label = ndimage.rotate(label, _rot, reshape=False).copy()
        return input, label, image_path
