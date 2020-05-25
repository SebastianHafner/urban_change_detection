import torch
from torch.utils import data as torch_data
from torchvision import transforms
from pathlib import Path
import numpy as np
import augmentations as aug

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
        available_features_sentinel2 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10',
                                        'B11', 'B12']
        selected_features_sentinel2 = cfg.DATASET.SENTINEL2.BANDS
        self.s2_feature_selection = self._get_feature_selection(available_features_sentinel2,
                                                                selected_features_sentinel2)

    def __getitem__(self, index):

        city = self.cities[index]

        # randomly choosing an orbit for sentinel1
        orbit = np.random.choice(ORBITS[city])

        t1_img = self._get_sentinel_data(city, orbit, 't1')
        t2_img = self._get_sentinel_data(city, orbit, 't2')

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

    def _get_sentinel_data(self, city, s1_orbit, t):

        s2_file = self.root_dir / city / 'sentinel2' / f'sentinel2_{city}_{t}.npy'
        s1_file = self.root_dir / city / 'sentinel1' / f'sentinel1_{city}_{s1_orbit}_{t}.npy'

        if self.cfg.DATASET.MODE == 'optical':
            img = np.load(s2_file)[:, :, self.s2_feature_selection]
        elif self.cfg.DATASET.MODE == 'radar':
            img = np.load(s1_file)[:, :, self.s1_feature_selection]
        else:  # fusion
            s1_img = np.load(s1_file)[:, :, self.s1_feature_selection]
            s2_img = np.load(s2_file)[:, :, self.s2_feature_selection]
            img = np.concatenate((s1_img, s2_img), axis=2)

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

    def sampler(self):
        if self.cfg.AUGMENTATION.OVERSAMPLING == 'pixel':
            sampling_weights = np.array([float(self._get_label_data(city).size) for city in self.cities])
        if self.cfg.AUGMENTATION.OVERSAMPLING == 'change':
            sampling_weights = np.array([float(np.sum(self._get_label_data(city))) for city in self.cities])
        sampler = torch_data.WeightedRandomSampler(weights=sampling_weights, num_samples=self.length,
                                                   replacement=True)
        return sampler
