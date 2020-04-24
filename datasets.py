import torch
from torchvision import transforms
from pathlib import Path
import numpy as np
import augmentations as aug
import utils


class OneraDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset: str, transform: list = None):
        super().__init__()

        self.cfg = cfg
        self.root_dir = Path(cfg.DATASET.PATH)

        if dataset == 'train':
            self.cities = [city for city in cfg.DATASET.ALL_CITIES if city not in cfg.DATASET.TEST_CITIES]
        else:
            self.cities = cfg.DATASET.TEST_CITIES

        self.length = len(self.cities)

        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([aug.Numpy2Torch()])

        self.mode = cfg.DATASET.MODE

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        available_features_sentinel1 = ['VV', 'VH']
        selected_features_sentinel1 = cfg.DATASET.SENTINEL1.BANDS
        self.s1_feature_selection = self._get_feature_selection(available_features_sentinel1,
                                                                selected_features_sentinel1)
        available_features_sentinel2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        selected_features_sentinel2 = cfg.DATASET.SENTINEL2.BANDS
        self.s2_feature_selection = self._get_feature_selection(available_features_sentinel2,
                                                                selected_features_sentinel2)

    def __getitem__(self, index):

        city = self.cities[index]

        img = self._get_sentinel_data(city)
        label = self._get_label_data(city)
        img, label = self.transform((img, label))

        sample = {
            'img': img,
            'label': label,
            'city': city
        }

        return sample

    def _get_sentinel_data(self, city):

        s2_dir = self.root_dir / city / 'sentinel2'

        s2_pre_file = s2_dir / f'sentinel2_{city}_pre.tif'
        pre, _, _ = utils.read_tif(s2_pre_file)

        s2_post_file = s2_dir / f'sentinel2_{city}_post.tif'
        post, _, _ = utils.read_tif(s2_post_file)

        img = np.concatenate([pre, post], axis=-1)

        return np.nan_to_num(img).astype(np.float32)

    def _get_label_data(self, city):

        label_file = self.root_dir / city / 'label' / f'urbanchange_{city}.tif'
        img, _, _ = utils.read_tif(label_file)

        return np.nan_to_num(img).astype(np.float32)

    def _get_feature_selection(self, features, selection):
        feature_selection = [False for _ in range(len(features))]
        for feature in selection:
            i = features.index(feature)
            feature_selection[i] = True
        return feature_selection

    def __len__(self):
        return self.length
