import rasterio
from typing import Tuple, Any
import numpy as np

def open_tif(tif_path: str) -> Tuple[np.ndarray, Any, Any, int, int]:
    """
    Opens a TIFF file and extracts its imagery data, CRS, transform, width, and height.

    Returns:
    - Tuple containing:
        - bands: numpy array representing the imagery data.
        - crs: coordinate reference system of the TIFF file.
        - transform: affine transform of the TIFF file.
        - width: int, the width of the TIFF file.
        - height: int, the height of the TIFF file.
    """
    with rasterio.open(tif_path) as src:
        bands = src.read()
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    return (bands, crs, transform, width, height)


def scale_imagery_bands(bands: np.ndarray, start_index: int = 2, scaling_factor: float = 10000.0) -> np.ndarray:
    """
    Scales the bands of the imagery starting from a given index by a scaling factor.

    Returns:
    - numpy array with scaled bands.
    """
    bands[start_index:] = bands[start_index:, :, :] / scaling_factor
    return bands

def normalize_bands(bands: np.ndarray) -> np.ndarray:
    """
    Normalizes the bands of the imagery to a range of 0 to 1.

    Returns:
    - numpy array with normalized bands.
    """
    min_vals = bands.min(axis=(1, 2), keepdims=True)
    max_vals = bands.max(axis=(1, 2), keepdims=True)
    normalized_bands = (bands - min_vals) / (max_vals - min_vals)
    return normalized_bands

def preprocess_bands(bands: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Preprocesses the imagery bands by scaling and normalizing them.

    Args:
    - bands: numpy array of imagery bands.
    - *args: Positional arguments for scale_imagery_bands.
    - **kwargs: Keyword arguments for scale_imagery_bands.

    Returns:
    - numpy array with preprocessed bands (scaled and normalized).
    """
    scaled_bands = scale_imagery_bands(bands, *args, **kwargs)
    normalized_bands = normalize_bands(scaled_bands)
    return normalized_bands





if __name__ == '__main__':
    tif_path = r"data\sample\Bramaputra_example_bands.tif"
    
    result = open_tif(tif_path)
    bands: np.ndarray = result[0]

    preprecessed_bands:np.ndarray = preprocess_bands(bands)
    print(preprecessed_bands.shape)
    