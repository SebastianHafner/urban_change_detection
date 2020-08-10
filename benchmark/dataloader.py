# PyTorch
import torch
import torchvision.transforms as tr

# Other
import os
import numpy as np
import random
from skimage import io
from scipy.ndimage import zoom
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import floor, ceil, sqrt, exp
from pathlib import Path

NORMALISE_IMGS = True
TYPE = 3  # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands
FP_MODIFIER = 10  # Tuning parameter, use 1 if unsure
DATA_AUG = True


class ChangeDetectionDataset(torch.utils.data.Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train=True, patch_side=96, stride=None, use_all_bands=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # basics
        if DATA_AUG:
            self.data_transform = tr.Compose([RandomFlip(), RandomRot()])
        else:
            self.data_transform = None
        self.root_path = path
        self.path_images = path / 'images'
        self.path_labels = path / 'labels'

        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride

        filename = 'train.txt' if train else 'test.txt'
        file = self.path_images / filename
        self.names = read_csv(str(file)).columns
        self.n_imgs = self.names.shape[0]

        n_pix = 0
        true_pix = 0

        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            I1, I2, cm = self.read_sentinel_img_trio(im_name)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm

            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()

            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i

            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_name,
                                            [self.stride * i, self.stride * i + self.patch_side, self.stride * j,
                                             self.stride * j + self.patch_side],
                                            [self.stride * (i + 1), self.stride * (j + 1)])
                    self.patch_coords.append(current_patch_coords)

        self.weights = [FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]

        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]

        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1 * np.array(label)).float()

        sample = {'I1': I1, 'I2': I2, 'label': label}

        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return sample

    def read_sentinel_img_trio(self, im_name: str):
        """Read cropped Sentinel-2 image pair and change map."""
        #     read images
        path_img1 = self.path_images / im_name / 'imgs_1'
        path_img2 = self.path_images / im_name / 'imgs_2'

        if TYPE == 0:
            I1 = read_sentinel_img(path_img1)
            I2 = read_sentinel_img(path_img2)
        elif TYPE == 1:
            I1 = read_sentinel_img_4(path_img1)
            I2 = read_sentinel_img_4(path_img2)
        elif TYPE == 2:
            I1 = read_sentinel_img_leq20(path_img1)
            I2 = read_sentinel_img_leq20(path_img2)
        elif TYPE == 3:
            I1 = read_sentinel_img_leq60(path_img1)
            I2 = read_sentinel_img_leq60(path_img2)

        label_file = self.path_labels / im_name / 'cm' / 'cm.png'
        cm = io.imread(str(label_file), as_gray=True) != 0

        # crop if necessary
        s1 = I1.shape
        s2 = I2.shape
        I2 = np.pad(I2, ((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0, 0)), 'edge')

        return I1, I2, cm


def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""

    # crop if necesary
    I = I[:s[0], :s[1]]
    si = I.shape

    # pad if necessary
    p0 = max(0, s[0] - si[0])
    p1 = max(0, s[1] - si[1])

    return np.pad(I, ((0, p0), (0, p1)), 'edge')


def read_sentinel_img(path: Path):
    """Read cropped Sentinel-2 image: RGB bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path / im_name / "B04.tif")
    g = io.imread(path / im_name / "B03.tif")
    b = io.imread(path / im_name / "B02.tif")

    I = np.stack((r, g, b), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_4(path: Path):
    """Read cropped Sentinel-2 image: RGB and NIR bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path / im_name / "B04.tif")
    g = io.imread(path / im_name / "B03.tif")
    b = io.imread(path / im_name / "B02.tif")
    nir = io.imread(path / im_name / "B08.tif")

    I = np.stack((r, g, b, nir), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_leq20(path: Path):
    """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path / im_name + "B04.tif")
    s = r.shape
    g = io.imread(path / im_name / "B03.tif")
    b = io.imread(path / im_name / "B02.tif")
    nir = io.imread(path / im_name / "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path / im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path / im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path / im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path / im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path / im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path / im_name + "B12.tif"), 2), s)

    I = np.stack((r, g, b, nir, ir1, ir2, ir3, nir2, swir2, swir3), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_leq60(path: Path):
    """Read cropped Sentinel-2 image: all bands."""
    im_name = os.listdir(path)[0][:-7]
    file = str(path / im_name)

    r = io.imread(file + "B04.tif")
    s = r.shape
    g = io.imread(file + "B03.tif")
    b = io.imread(file + "B02.tif")
    nir = io.imread(file + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(file + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(file + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(file + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(file + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(file + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(file + "B12.tif"), 2), s)

    uv = adjust_shape(zoom(io.imread(file + "B01.tif"), 6), s)
    wv = adjust_shape(zoom(io.imread(file + "B09.tif"), 6), s)
    swirc = adjust_shape(zoom(io.imread(file + "B10.tif"), 6), s)

    I = np.stack((r, g, b, nir, ir1, ir2, ir3, nir2, swir2, swir3, uv, wv, swirc), axis=2).astype('float')

    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I





def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    #     out = np.swapaxes(I,1,2)
    #     out = np.swapaxes(out,0,1)
    #     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)


class RandomFlip(object):
    """Flip randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        if random.random() > 0.5:
            I1 = I1.numpy()[:, :, ::-1].copy()
            I1 = torch.from_numpy(I1)
            I2 = I2.numpy()[:, :, ::-1].copy()
            I2 = torch.from_numpy(I2)
            label = label.numpy()[:, ::-1].copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']

        n = random.randint(0, 3)
        if n:
            I1 = sample['I1'].numpy()
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I1 = torch.from_numpy(I1)
            I2 = sample['I2'].numpy()
            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            I2 = torch.from_numpy(I2)
            label = sample['label'].numpy()
            label = np.rot90(label, n, axes=(0, 1)).copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}