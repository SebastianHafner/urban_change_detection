import numpy as np
from pathlib import Path
import utils
import cv2


def get_band(file: Path) -> str:
    return file.stem.split('_')[-1]


def combine_bands(folder: Path) -> tuple:

    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    n_bands = len(bands)

    data = {get_band(file): file for file in folder.glob('**/*')}

    # using blue band as reference (10 m)
    blue, transform, crs = utils.read_tif(data['B02'])
    h, w, _ = blue.shape

    img = np.ndarray((h, w, n_bands), dtype=np.float32)
    for i, band in enumerate(bands):
        arr, _, _ = utils.read_tif(data[band])
        arr = arr[:, :, 0]
        band_h, band_w = arr.shape

        # up-sample 20 m bands
        if not band_h == h and not band_w == w:
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)

        arr = np.clip(arr / 10000, a_min=0, a_max=1)
        img[:, :, i] = arr

    return img, transform, crs


def process_city(img_folder: Path, label_folder: Path, city: str, new_root: Path) -> None:

    print(city)

    new_parent = new_root / city
    new_parent.mkdir(exist_ok=True)

    # image data
    for pre_post in ['pre', 'post']:

        # get data
        from_folder = img_folder / city / 'imgs_1' if pre_post == 'pre' else img_folder / city / 'imgs_2'
        img, transform, crs = combine_bands(from_folder)

        # save data
        to_folder = new_parent / 'sentinel2'
        to_folder.mkdir(exist_ok=True)
        save_file = to_folder / f'sentinel2_{city}_{pre_post}.tif'
        utils.write_tif(save_file, img, transform, crs)

    from_label_file = label_folder / city / 'cm' / f'{city}-cm.tif'
    label, _, _ = utils.read_tif(from_label_file)
    label = label - 1

    to_label_file = new_parent / 'label' / f'urbanchange_{city}.tif'
    to_label_file.parent.mkdir(exist_ok=True)
    utils.write_tif(to_label_file, label, transform, crs)


if __name__ == '__main__':
    # assume unchanged Onera dataset
    IMG_FOLDER = Path('C:/Users/hafne/urban_change_detection/data/Onera/images/')
    LABEL_FOLDER = Path('C:/Users/hafne/urban_change_detection/data/Onera/labels/')
    NEW_ROOT = Path('C:/Users/hafne/urban_change_detection/data/Onera/preprocessed/')

    cities = ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'cupertino', 'hongkong',
              'mumbai', 'nantes', 'paris', 'pisa', 'rennes', 'saclay_e']

    for city in cities:
        process_city(IMG_FOLDER, LABEL_FOLDER, city, NEW_ROOT)
