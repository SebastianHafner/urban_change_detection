import torch
from torchvision import transforms
from pathlib import Path
import numpy as np
import augmentations as aug


class OSCDDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset: str, no_augmentation: bool = False):
        super().__init__()

        self.cfg = cfg
        self.root_dir = Path(cfg.DATASET.PATH)

        if dataset == 'train':
            self.cities = cfg.DATASET.TRAIN
        else:
            self.cities = cfg.DATASET.TEST

        self.length = len(self.cities)

        if no_augmentation:
            self.transform = transforms.Compose([aug.Numpy2Torch()])
        else:
            self.transform = aug.compose_transformations(cfg)

        self.mode = cfg.DATASET.MODE

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        available_features_sentinel1 = ['VV']
        selected_features_sentinel1 = cfg.DATASET.SENTINEL1.BANDS
        self.s1_feature_selection = self._get_feature_selection(available_features_sentinel1,
                                                                selected_features_sentinel1)
        available_features_sentinel2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        selected_features_sentinel2 = cfg.DATASET.SENTINEL2.BANDS
        self.s2_feature_selection = self._get_feature_selection(available_features_sentinel2,
                                                                selected_features_sentinel2)

    def __getitem__(self, index):

        city = self.cities[index]

        t1_img = self._get_sentinel_data(city, 't1')
        t2_img = self._get_sentinel_data(city, 't2')

        label = self._get_label_data(city)
        label = label[:, :, np.newaxis]

        t1_img, t2_img, label = self.transform((t1_img, t2_img, label))
        sample = {
            't1_img': t1_img,
            't2_img': t2_img,
            'label': label,
            'city': city
        }

        return sample

    def _get_sentinel_data(self, city, t):

        s2_file = self.root_dir / city / 'sentinel2' / f'sentinel2_{city}_{t}.npy'
        s1_file = self.root_dir / city / 'sentinel1' / f'sentinel1_{city}_{t}.npy'

        if self.cfg.DATASET.MODE == 'optical':
            img = np.load(s2_file)
        elif self.cfg.DATASET.MODE == 'radar':
            img = np.load(s1_file)
        else:  # fusion
            s1_img = np.load(s1_file)
            s2_img = np.load(s2_file)
            img = np.concatenate((s1_img, s2_img), axis=2)
        # TODO: subset according to band selection

        return img.astype(np.float32)

    def _get_label_data(self, city):
        label_file = self.root_dir / city / 'label' / f'urbanchange_{city}.npy'
        label = np.load(label_file).astype(np.float32)
        return label

    def _get_feature_selection(self, features, selection):
        feature_selection = [False for _ in range(len(features))]
        for feature in selection:
            i = features.index(feature)
            feature_selection[i] = True
        return feature_selection

    def __len__(self):
        return self.length
