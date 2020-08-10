import torch
from torch.utils import data as torch_data
from torchvision import transforms
from pathlib import Path
import numpy as np
import augmentations as aug
from utils import *
from tqdm import tqdm as tqdm
import pandas as pd
import math
from skimage import io
from scipy.ndimage import zoom
import os
import random

ORBITS = {
    'aguasclaras': [24],
    'bercy': [59, 8, 110],
    'bordeaux': [30, 8, 81],
    'nantes': [30, 81],
    'paris': [59, 8, 110],
    'rennes': [30, 81],
    'saclay_e': [59, 8],
    'abudhabi': [130],
    'cupertino': [35, 115, 42],
    'pisa': [15, 168],
    'beihai': [157],
    'hongkong': [11, 113],
    'beirut': [14, 87],
    'mumbai': [34],
    'brasilia': [24],
    'montpellier': [59, 37],
    'norcia': [117, 44, 22, 95],
    'rio': [155],
    'saclay_w': [59, 8, 110],
    'valencia': [30, 103, 8, 110],
    'dubai': [130, 166],
    'lasvegas': [166, 173],
    'milano': [66, 168],
    'chongqing': [55, 164]
}

SENTINEL1_BANDS = ['VV']
SENTINEL2_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']


class OSCDDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset: str, no_augmentation: bool = False):
        super().__init__()

        self.cfg = cfg
        self.root_dir = Path(cfg.DATASET.PATH)

        if dataset == 'train':
            multiplier = cfg.DATASET.TRAIN_MULTIPLIER
            self.cities = multiplier * cfg.DATASET.TRAIN
        else:
            self.cities = cfg.DATASET.TEST

        self.length = len(self.cities)

        if no_augmentation:
            self.transform = transforms.Compose([aug.Numpy2Torch()])
        else:
            self.transform = aug.compose_transformations(cfg)

        self.mode = cfg.DATASET.MODE

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_band_selection = self.get_band_selection(SENTINEL1_BANDS, cfg.DATASET.SENTINEL1_BANDS)
        self.s2_band_selection = self.get_band_selection(SENTINEL2_BANDS, cfg.DATASET.SENTINEL2_BANDS)

    def __getitem__(self, index):

        city = self.cities[index]

        # randomly choosing an orbit for sentinel1
        orbit = np.random.choice(ORBITS[city])
        # orbit = ORBITS[city][0]

        if self.cfg.DATASET.MODE == 'optical':
            t1_img = self._get_sentinel2_data(city, 't1')
            t2_img = self._get_sentinel2_data(city, 't2')
        elif self.cfg.DATASET.MODE == 'sar':
            t1_img = self._get_sentinel1_data(city, orbit, 't1')
            t2_img = self._get_sentinel1_data(city, orbit, 't2')
        else:
            s1_t1_img = self._get_sentinel1_data(city, orbit, 't1')
            s2_t1_img = self._get_sentinel2_data(city, 't1')
            t1_img = np.concatenate((s1_t1_img, s2_t1_img), axis=2)

            s1_t2_img = self._get_sentinel1_data(city, orbit, 't2')
            s2_t2_img = self._get_sentinel2_data(city, 't2')
            t2_img = np.concatenate((s1_t2_img, s2_t2_img), axis=2)

        label = self._get_label_data(city)
        t1_img, t2_img, label = self.transform((t1_img, t2_img, label))

        sample = {
            't1_img': t1_img,
            't2_img': t2_img,
            'label': label,
            'city': city
        }

        return sample

    def _get_sentinel1_data(self, city, orbit, t):
        file = self.root_dir / city / 'sentinel1' / f'sentinel1_{city}_{orbit}_{t}.npy'
        img = np.load(file)[:, :, self.s1_band_selection]
        return img.astype(np.float32)

    def _get_sentinel2_data(self, city, t):
        file = self.root_dir / city / 'sentinel2' / f'sentinel2_{city}_{t}.npy'
        img = np.load(file)[:, :, self.s2_band_selection]
        return img.astype(np.float32)

    def _get_label_data(self, city):
        label_file = self.root_dir / city / 'label' / f'urbanchange_{city}.npy'
        label = np.load(label_file).astype(np.float32)
        label = label[:, :, np.newaxis]
        return label

    def __len__(self):
        return self.length

    def band_index(self, band: str) -> int:
        s1_bands = self.cfg.DATASET.SENTINEL1_BANDS
        s2_bands = self.cfg.DATASET.SENTINEL2_BANDS
        mode = self.cfg.DATASET.MODE

        index = s1_bands.index(band) if band in s1_bands else s2_bands.index(band)

        # handle band concatenation for fusion
        if mode == 'fusion' and band in s2_bands:
            index += len(s1_bands)

        return index

    def sampler(self):
        if self.cfg.AUGMENTATION.OVERSAMPLING == 'pixel':
            sampling_weights = np.array([float(self._get_label_data(city).size) for city in self.cities])
        if self.cfg.AUGMENTATION.OVERSAMPLING == 'change':
            sampling_weights = np.array([float(np.sum(self._get_label_data(city))) for city in self.cities])
        sampler = torch_data.WeightedRandomSampler(weights=sampling_weights, num_samples=self.length,
                                                   replacement=True)
        return sampler


class OSCDDatasetFast(torch.utils.data.Dataset):
    def __init__(self, cfg, run_type: str, no_augmentation: bool = False):
        super().__init__()

        self.cfg = cfg
        self.root_dir = Path(cfg.DATASET.PATH)

        self.img_names = cfg.DATASET.TRAIN if run_type == 'train' else cfg.DATASET.TEST
        self.imgs_dataset = self.img_names * cfg.DATASET.TRAIN_MULTIPLIER if run_type == 'train' else self.img_names

        self.length = len(self.imgs_dataset)

        if no_augmentation:
            self.transform = transforms.Compose([aug.Numpy2Torch()])
        else:
            self.transform = aug.compose_transformations(cfg)

        self.mode = cfg.DATASET.MODE

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_band_selection = get_band_selection(SENTINEL1_BANDS, cfg.DATASET.SENTINEL1_BANDS)
        self.s2_band_selection = get_band_selection(SENTINEL2_BANDS, cfg.DATASET.SENTINEL2_BANDS)

        # loading data into memory
        self.s1_t1_imgs, self.s1_t2_imgs = {}, {}
        self.s2_t1_imgs, self.s2_t2_imgs = {}, {}
        self.label_imgs = {}
        for img_name in tqdm(self.img_names):

            s1_t1_orbits, s1_t2_orbits = {}, {}
            for orbit in ORBITS[img_name]:
                s1_t1_img = self._get_sentinel1_data(img_name, orbit, 't1')
                s1_t2_img = self._get_sentinel1_data(img_name, orbit, 't2')
                s1_t1_orbits[orbit] = s1_t1_img
                s1_t2_orbits[orbit] = s1_t2_img
            self.s1_t1_imgs[img_name] = s1_t1_orbits
            self.s1_t2_imgs[img_name] = s1_t2_orbits

            self.s2_t1_imgs[img_name] = self._get_sentinel2_data(img_name, 't1')
            self.s2_t2_imgs[img_name] = self._get_sentinel2_data(img_name, 't2')

            self.label_imgs[img_name] = self._get_label_data(img_name)

    def __getitem__(self, index):

        img_name = self.imgs_dataset[index]
        t1_img, t2_img, label = self._get_img_triple(img_name)
        t1_img, t2_img, label = self.transform((t1_img, t2_img, label))

        sample = {
            't1_img': t1_img,
            't2_img': t2_img,
            'label': label,
            'city': img_name
        }

        return sample

    def get_evaluation_triple(self, img_name):
        t1_img, t2_img, label = self._get_img_triple(img_name)
        to_torch = aug.Numpy2Torch()
        t1_img, t2_img, label = to_torch((t1_img, t2_img, label))
        return t1_img, t2_img, label

    def _get_img_triple(self, img_name):
        # randomly choosing an orbit for sentinel1
        orbit = np.random.choice(ORBITS[img_name])

        if self.cfg.DATASET.MODE == 'optical':
            t1_img = self.s2_t1_imgs[img_name]
            t2_img = self.s2_t2_imgs[img_name]
        elif self.cfg.DATASET.MODE == 'sar':
            t1_img = self.s1_t1_imgs[img_name][orbit]
            t2_img = self.s1_t2_imgs[img_name][orbit]
        else:
            s1_t1_img = self.s1_t1_imgs[img_name][orbit]
            s2_t1_img = self.s2_t1_imgs[img_name]
            t1_img = np.concatenate((s1_t1_img, s2_t1_img), axis=2)

            s1_t2_img = self.s1_t2_imgs[img_name][orbit]
            s2_t2_img = self.s2_t2_imgs[img_name]
            t2_img = np.concatenate((s1_t2_img, s2_t2_img), axis=2)

        label = self.label_imgs[img_name]

        return t1_img, t2_img, label

    def _get_sentinel1_data(self, city, orbit, t):
        file = self.root_dir / city / 'sentinel1' / f'sentinel1_{city}_{orbit}_{t}.npy'
        img = np.load(file)[:, :, self.s1_band_selection]
        return img.astype(np.float32)

    def _get_sentinel2_data(self, city, t):
        file = self.root_dir / city / 'sentinel2' / f'sentinel2_{city}_{t}.npy'
        img = np.load(file)[:, :, self.s2_band_selection]
        return img.astype(np.float32)

    def _get_label_data(self, city):
        label_file = self.root_dir / city / 'label' / f'urbanchange_{city}.npy'
        label = np.load(label_file).astype(np.float32)
        label = label[:, :, np.newaxis]
        return label

    def __len__(self):
        return self.length

    def band_index(self, band: str) -> int:
        s1_bands = self.cfg.DATASET.SENTINEL1_BANDS
        s2_bands = self.cfg.DATASET.SENTINEL2_BANDS
        mode = self.cfg.DATASET.MODE

        index = s1_bands.index(band) if band in s1_bands else s2_bands.index(band)

        # handle band concatenation for fusion
        if mode == 'fusion' and band in s2_bands:
            index += len(s1_bands)

        return index

    def sampler(self):
        if self.cfg.AUGMENTATION.OVERSAMPLING == 'pixel':
            sampling_weights = np.array([float(self._get_label_data(city).size) for city in self.cities])
        if self.cfg.AUGMENTATION.OVERSAMPLING == 'change':
            sampling_weights = np.array([float(np.sum(self._get_label_data(city))) for city in self.cities])
        sampler = torch_data.WeightedRandomSampler(weights=sampling_weights, num_samples=self.length,
                                                   replacement=True)
        return sampler


class OSCDDifferenceImages(OSCDDataset):

    def __init__(self, cfg, dataset: str, no_augmentation: bool = False):
        super().__init__(cfg, dataset, no_augmentation)

    def __getitem__(self, index):

        city = self.cities[index]

        # randomly choosing an orbit for sentinel1
        orbit = np.random.choice(ORBITS[city])

        if self.mode == 'optical':
            t1_img = self._get_sentinel2_data(city, 't1')
            t2_img = self._get_sentinel2_data(city, 't2')
            diff_img = self.optical_difference_image(t1_img, t2_img)
        elif self.mode == 'sar':
            t1_img = self._get_sentinel1_data(city, orbit, 't1')
            t2_img = self._get_sentinel1_data(city, orbit, 't2')
            diff_img = self.sar_difference_image(t1_img, t2_img)
        else:
            s1_t1_img = self._get_sentinel1_data(city, orbit, 't1')
            s1_t2_img = self._get_sentinel1_data(city, orbit, 't2')
            s1_diff_img = self.sar_difference_image(s1_t1_img, s1_t2_img)

            s2_t1_img = self._get_sentinel2_data(city, 't1')
            s2_t2_img = self._get_sentinel2_data(city, 't2')
            s2_diff_img = self.optical_differece_image(s2_t1_img, s2_t2_img)

            diff_img = np.concatenate((s1_diff_img, s2_diff_img), axis=2)

        label = self._get_label_data(city)
        diff_img, _, label = self.transform((diff_img, diff_img, label))

        sample = {
            'diff_img': diff_img,
            'label': label,
            'city': city
        }

        return sample

    def optical_difference_image(self, t1_img: np.ndarray, t2_img: np.ndarray) -> np.ndarray:

        if self.cfg.DATASET.INDICES:
            # change vector analysis with ndvi and ndbi
            red = self.band_index('B04')
            nir = self.band_index('B08')
            swir = self.band_index('B11')

            t1_ndvi = normalize(normalized_difference(t1_img, red, nir), -1, 1)
            t1_ndbi = normalize(normalized_difference(t1_img, swir, nir), -1, 1)
            t1_indices = np.stack((t1_ndvi, t1_ndbi), axis=-1)

            t2_ndvi = normalize(normalized_difference(t2_img, red, nir), -1, 1)
            t2_ndbi = normalize(normalized_difference(t2_img, swir, nir), -1, 1)
            t2_indices = np.stack((t2_ndvi, t2_ndbi), axis=-1)
            t1_img, t2_img = t1_indices, t2_indices

        cva = change_vector_analysis(t1_img, t2_img)

        return cva.astype(np.float32)

    def sar_difference_image(self, t1_img: np.ndarray, t2_img: np.ndarray) -> np.ndarray:

        # log ration of VV
        vv = self.band_index('VV')

        t1_vv = t1_img[:, :, vv]
        t2_vv = t2_img[:, :, vv]

        lr = log_ratio(t1_vv, t2_vv)
        # TODO: add normalization
        return lr.astype(np.float32)


class OSCDDatasetPaper(torch.utils.data.Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, root_path: Path, train: bool = True, patch_side: int = 96, stride=None,
                 augmentation: bool = False, normalize: bool = False, fp_modifier: int = 1, use_all_bands=False):

        # basics
        self.root_path = root_path
        self.path_images = root_path / 'images'
        self.path_labels = root_path / 'labels'

        self.data_transform = transforms.Compose([RandomFlip(), RandomRot()]) if augmentation else None
        self.normalize = normalize
        self.fp_modifier = fp_modifier

        self.patch_side = patch_side
        self.stride = 1 if stride is None else stride

        filename = 'train.txt' if train else 'test.txt'
        file = self.path_images / filename
        self.names = pd.read_csv(str(file)).columns
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
            n1 = math.ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = math.ceil((s[2] - self.patch_side + 1) / self.stride)
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


        self.weights = [self.fp_modifier * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

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

        sample = {'t1_img': I1, 't2_img': I2, 'label': label}

        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return sample

    def read_sentinel_img_trio(self, im_name: str):
        """Read cropped Sentinel-2 image pair and change map."""
        #     read images
        path_img1 = self.path_images / im_name / 'imgs_1'
        path_img2 = self.path_images / im_name / 'imgs_2'

        t1_img = self.read_sentinel_img_leq60(path_img1)
        t2_img = self.read_sentinel_img_leq60(path_img2)

        label_file = self.path_labels / im_name / 'cm' / 'cm.png'
        label = io.imread(str(label_file), as_gray=True) != 0

        # crop if necessary
        s1 = t1_img.shape
        s2 = t2_img.shape
        t2_img = np.pad(t2_img, ((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0, 0)), 'edge')

        return t1_img, t2_img, label

    def read_sentinel_img_leq60(self, path: Path):
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

        if self.normalize:
            I = (I - I.mean()) / I.std()

        return I


class OSCDDatasetNewPrep(torch.utils.data.Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, cfg, dataset: str, no_augmentation: bool = False):

        # basics
        self.cfg = cfg

        self.root_path = Path(cfg.DATASET.PATH).parent
        self.path_images = self.root_path / 'images'
        self.path_labels = self.root_path / 'labels'

        self.data_transform = None if no_augmentation else transforms.Compose([RandomFlip(), RandomRot()])

        file = self.path_images / f'{dataset}.txt'
        self.names = pd.read_csv(str(file)).columns
        self.n_imgs = self.names.shape[0]

        self.normalize = True
        self.fp_modifier = 1
        self.patch_side = 96
        self.stride = 1

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_band_selection = get_band_selection(SENTINEL1_BANDS, cfg.DATASET.SENTINEL1_BANDS)
        self.s2_band_selection = get_band_selection(SENTINEL2_BANDS, cfg.DATASET.SENTINEL2_BANDS)

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
            n1 = math.ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = math.ceil((s[2] - self.patch_side + 1) / self.stride)
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

        self.weights = [self.fp_modifier * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, img_name: str):

        t1_img = self.imgs_1[img_name][self.s2_band_selection, ]
        t2_img = self.imgs_2[img_name][self.s2_band_selection, ]
        cm = self.change_maps[img_name]
        cm = transforms.functional.to_tensor(cm)

        return t1_img, t2_img, cm

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]

        I1 = self.imgs_1[im_name]
        I1 = I1[self.s2_band_selection, limits[0]:limits[1], limits[2]:limits[3]]

        I2 = self.imgs_2[im_name]
        I2 = I2[self.s2_band_selection, limits[0]:limits[1], limits[2]:limits[3]]

        label = self.change_maps[im_name]
        label = label[limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1 * np.array(label)).float()

        sample = {'t1_img': I1, 't2_img': I2, 'label': label}

        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return sample

    def read_sentinel_img_trio(self, im_name: str):
        """Read cropped Sentinel-2 image pair and change map."""
        #     read images
        path_img1 = self.path_images / im_name / 'imgs_1'
        path_img2 = self.path_images / im_name / 'imgs_2'

        t1_img = self.read_sentinel_img_leq60(path_img1)
        t2_img = self.read_sentinel_img_leq60(path_img2)

        label_file = self.path_labels / im_name / 'cm' / 'cm.png'
        label = io.imread(str(label_file), as_gray=True) != 0

        # crop if necessary
        s1 = t1_img.shape
        s2 = t2_img.shape
        t2_img = np.pad(t2_img, ((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0, 0)), 'edge')

        return t1_img, t2_img, label

    def read_sentinel_img_leq60(self, path: Path):
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

        if self.normalize:
            I = (I - I.mean()) / I.std()

        return I.astype(np.float32)


def get_band_selection(bands: list, selection: list):
    bands_selection = [False for _ in range(len(bands))]
    for band in selection:
        i = bands.index(band)
        bands_selection[i] = True
    return bands_selection


# Adjust shape of grayscale image I to s.
def adjust_shape(I, s):
    # crop if necessary
    I = I[:s[0], :s[1]]
    si = I.shape

    # pad if necessary
    p0 = max(0, s[0] - si[0])
    p1 = max(0, s[1] - si[1])

    return np.pad(I, ((0, p0), (0, p1)), 'edge')


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
        t1_img, t2_img, label = sample['t1_img'], sample['t2_img'], sample['label']

        if random.random() > 0.5:
            t1_img = t1_img.numpy()[:, :, ::-1].copy()
            t1_img = torch.from_numpy(t1_img)
            t2_img = t2_img.numpy()[:, :, ::-1].copy()
            t2_img = torch.from_numpy(t2_img)
            label = label.numpy()[:, ::-1].copy()
            label = torch.from_numpy(label)

        return {'t1_img': t1_img, 't2_img': t2_img, 'label': label}


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        t1_img, t2_img, label = sample['t1_img'], sample['t2_img'], sample['label']

        n = random.randint(0, 3)
        if n:
            t1_img = sample['t1_img'].numpy()
            t1_img = np.rot90(t1_img, n, axes=(1, 2)).copy()
            t1_img = torch.from_numpy(t1_img)
            t2_img = sample['t2_img'].numpy()
            t2_img = np.rot90(t2_img, n, axes=(1, 2)).copy()
            t2_img = torch.from_numpy(t2_img)
            label = sample['label'].numpy()
            label = np.rot90(label, n, axes=(0, 1)).copy()
            label = torch.from_numpy(label)

        return {'t1_img': t1_img, 't2_img': t2_img, 'label': label}
