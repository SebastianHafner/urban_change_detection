import numpy as np
from pathlib import Path
import utils
import cv2
import tifffile


def get_band(file: Path) -> str:
    return file.stem.split('_')[-1]


def combine_bands(folder: Path) -> tuple:

    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    n_bands = len(bands)

    # using blue band as reference (10 m) to create img
    blue_file = folder / 'B02.tif'
    blue = tifffile.imread(str(blue_file))
    img = np.ndarray((*blue.shape, n_bands), dtype=np.float32)

    for i, band in enumerate(bands):
        band_file = folder / f'{band}.tif'
        arr = tifffile.imread(str(band_file))
        band_h, band_w = arr.shape

        # up-sample 20 m bands
        # arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)

        # rescaling image to [0, 1]
        arr = np.clip(arr / 10000, a_min=0, a_max=1)
        img[:, :, i] = arr

    return img


def process_city(img_folder: Path, label_folder: Path, city: str, new_root: Path) -> None:

    print(city)

    new_parent = new_root / city
    new_parent.mkdir(exist_ok=True)

    # image data
    for pre_post in ['pre', 'post']:

        # get data
        from_folder = img_folder / city / 'imgs_1_rect' if pre_post == 'pre' else img_folder / city / 'imgs_2_rect'
        img = combine_bands(from_folder)

        # save data
        to_folder = new_parent / 'sentinel2'
        to_folder.mkdir(exist_ok=True)

        save_file = to_folder / f'sentinel2_{city}_{pre_post}.npy'
        np.save(save_file, img)

    from_label_file = label_folder / city / 'cm' / f'{city}-cm.tif'
    label = tifffile.imread(str(from_label_file))
    label = label - 1

    to_label_file = new_parent / 'label' / f'urbanchange_{city}.npy'
    to_label_file.parent.mkdir(exist_ok=True)
    np.save(to_label_file, label)


if __name__ == '__main__':
    # assume unchanged OSCD dataset
    # IMG_FOLDER = Path('C:/Users/hafne/urban_change_detection/data/Onera/images/')
    IMG_FOLDER = Path('C:/Users/shafner/urban_change_detection/OSCD_dataset/images/')
    # LABEL_FOLDER = Path('C:/Users/hafne/urban_change_detection/data/Onera/labels/')
    LABEL_FOLDER = Path('C:/Users/shafner/urban_change_detection/OSCD_dataset/labels/')

    # NEW_ROOT = Path('C:/Users/hafne/urban_change_detection/data/Onera/preprocessed/')
    NEW_ROOT = Path('C:/Users/shafner/urban_change_detection/OSCD_dataset/preprocessed/')

    cities = ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'cupertino', 'hongkong',
              'mumbai', 'nantes', 'paris', 'pisa', 'rennes', 'saclay_e']

    for city in cities:
        process_city(IMG_FOLDER, LABEL_FOLDER, city, NEW_ROOT)
