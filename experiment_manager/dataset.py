import torch
import os
import numpy as np
import cv2

class SimpleInferenceDataset(torch.utils.data.Dataset):
    '''
    A dataset objects that lists
    '''
    def __init__(self, dataset_path, file_extension='.png', downsample_scale=None, filter=None):

        image_files = []

        for filename in os.listdir(dataset_path):
            if filename.endswith(file_extension) and filter in filename:
                image_files.append(filename)

        self.image_files = image_files
        self.length = len(image_files)
        self.dataset_path = dataset_path
        self.downsample_scale = downsample_scale
        self.filter = filter

    def __getitem__(self, index):
        label = np.zeros(1)
        image_filename = self.image_files[index]
        x = self._process_input(image_filename)
        return x, label, image_filename

    def _process_input(self, image_filename):
        img_path = os.path.join(self.dataset_path, image_filename)
        img = cv2.imread(img_path)

        if self.downsample_scale is not None:
            scale = self.downsample_scale
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # BGR to RGB
        img = img[...,::-1]

        img = img.astype(np.float32) / 255.
        # move from (x, y, c) to (c, x, y) PyTorch style
        img = np.moveaxis(img, -1, 0)

        return img


    def __len__(self):

        return self.length