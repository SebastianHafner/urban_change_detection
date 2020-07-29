import torch
import rasterio
from pathlib import Path
import math


# reading in geotiff file as numpy array
def read_tif(file: Path):

    if not file.exists():
        raise FileNotFoundError(f'File {file} not found')

    with rasterio.open(file) as dataset:
        arr = dataset.read()  # (bands X height X width)
        transform = dataset.transform
        crs = dataset.crs

    return arr.transpose((1, 2, 0)), transform, crs


# writing an array to a geo tiff file
def write_tif(file: Path, arr, transform, crs):

    if not file.parent.exists():
        file.parent.mkdir()

    height, width, bands = arr.shape
    with rasterio.open(
            file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(arr[:, :, i], i + 1)


def to_numpy(tensor:torch.Tensor):
    return tensor.cpu().detach().numpy()


def euclidean_distance(t1_img: torch.Tensor, t2_img: torch.Tensor) -> torch.Tensor:
    diff = t1_img - t2_img
    diff_squared = torch.square(diff)
    sum_ = torch.sum(diff_squared, dim=0, keepdim=True)
    distance = torch.sqrt(sum_)
    return distance


def spectral_angle_mapper(t1_img: torch.Tensor, t2_img: torch.Tensor) -> torch.Tensor:

    nominator = 0

    denominator1 = torch.sqrt(torch.sum(torch.square(t1_img), dim=0))
    denominator2 = torch.sqrt(torch.sum(torch.square(t2_img), dim=0))

    angle = torch.acos((nominator / (denominator1 + denominator2)))
    return angle


# according to formula in: https://reader.elsevier.com/reader/sd/pii/S1878029611008607?token=95EB0C4C341A3C82861FDB0F250C9315C15C62CF192A4A3823750BB21E22DBDCC17D19E07E019CF03FCA328AAEAA3718
def change_vector_analysis(t1_img: torch.Tensor, t2_img: torch.Tensor) -> tuple:
    # images with shape (C x H x W)

    # computing euclidean distance
    distance = euclidean_distance(t1_img, t2_img)

    # computing direction (angle)
    direction = spectral_angle_mapper(t1_img, t2_img)
    direction = normalize(direction, 0, math.pi)

    return distance, direction


def log_ratio(t1_img: torch.Tensor, t2_img: torch.Tensor) -> torch.Tensor:
    logR = t1_img - t2_img
    logR = normalize(logR, -1, 1, 0, 1)
    return logR


def rescale(img: torch.Tensor, from_min: float, from_max: float, to_min: float, to_max: float) -> torch.Tensor:
    return img


def normalize(img: torch.Tensor, from_min: float, from_max: float) -> torch.Tensor:
    return rescale(img, from_min, from_max, 0, 1)


if __name__ == '__main__':

    a = torch.ones(2, 4, 4).to('cuda')
    a[0, ] = 0
    a[1, ] = 0
    b = torch.ones(2, 4, 4).to('cuda')
    b[0, ] = 1
    b[1, ] = 1
    distance = euclidean_distance(a, b)
    print(distance)

    direction = spectral_angle_mapper(a, b)
    print(direction)

    pass
