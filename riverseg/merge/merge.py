from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.transform import from_origin, Affine

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import preprocess.patch_preprocessing as riverpp
import predict.predict as riverpred


def list_tif_paths(files_dir: str) -> list[str]:
    """
    Returns list of all tif files in a folder
    
    Args:
    - files_dir: Path to folder containing tif files

    Returns:
    - List of all tif files in the folder.
    """
    paths = glob.glob(files_dir + r"\*.tif")
    
    if not paths:
        raise FileNotFoundError(f"No .tif files found in : {files_dir}")
    return paths

def assert_same_crs(tif_paths: Iterable[str], expected_crs) -> None:
    """
    Controls that all tif files have the same expected crs (Coordinate Reference System)

    Args: 
    - tif_paths: list containing all tif files.
    - expected_crs: The correct reference system expected.

    Returns:
    - None
    """
    for path in tif_paths:
        with rasterio.open(path) as src:
            if src.crs != expected_crs:
                raise ValueError(f"CRS mismatch: {path} has crs {src.crs}, expected crs: {expected_crs}")

def compute_union_bounds(tif_paths: Iterable[str]) -> Tuple[float, float, float, float]:
    """
    Compute union bounds (left, bottom, right, top) across all rasters

    Args: 
    - tif_paths: list containing all tif files.
   
    Returns:
        A bounding box covering the spatial union of all input rasters,
        given as (left, bottom, right, top) in map coordinates (same CRS as the input rasters).
    """

    lefts, bottoms, rights, tops = [], [], [], []
    for p in tif_paths:
        with rasterio.open(p) as src:
            b = src.bounds
            lefts.append(b.left); bottoms.append(b.bottom); rights.append(b.right); tops.append(b.top)

    left, bottom, right, top = min(lefts), min(bottoms), max(rights), max(tops)
    return left, bottom, right, top

@dataclass(frozen=True)
class MosaicSpec:
    transform: Affine
    width: int
    height: int
    crs: rasterio.crs.CRS

def build_mosaic_spec(tif_paths: List[str]) -> MosaicSpec:
    """
    Create output grid (transform, width, height, crs) that covers all tiles.

    Args: 
    - tif_paths: list containing tif files.
   
    Returns:
        A bounding box covering the spatial union of all input rasters,
        given as (left, bottom, right, top) in map coordinates (same CRS as the input rasters).
    """
    with rasterio.open(tif_paths[0]) as src0:
        crs = src0.crs
        res_x = src0.transform.a
        res_y = -src0.transform.e  # positive pixel size

    assert_same_crs(tif_paths, crs)

    left, bottom, right, top = compute_union_bounds(tif_paths)

    transform = from_origin(left, top, res_x, res_y)
    width = int(np.ceil((right - left) / res_x))
    height = int(np.ceil((top - bottom) / res_y))

    return MosaicSpec(transform, width, height, crs)

def build_output_meta(spec: MosaicSpec, compress: str = "lzw") -> dict:
    """
    Create rasterio metadata for a single-band uint8 binary mask (0/1).

    Args;
    - spec: Mosaic specs as a MosaicSpec datastructure
    - compress: Compression method (default: "lzw")

    Returns:
        A dictionary containing the metadata for the output raster
    """
    return{
        "driver": "GTiff",
        "height": spec.height,
        "width": spec.width,
        "count": 1,
        "crs": spec.crs,
        "transform": spec.transform,
        "dtype": "uint8",
        "compress": compress,
    }


def squeeze_to_hw(pred: np.ndarray) -> np.ndarray:
    """
    Squeezes a prediction array to 2D (H, W) shape.

    This function converts input arrays of shape:
    - (B, C, H, W): Takes the first batch and first channel.
    - (C, H, W): Takes the first channel.
    - (H, W): Returns as is.

    Args:
    - pred (np.ndarray): Input prediction array of shape (B, C, H, W), (C, H, W), or (H, W).

    Returns:
    - np.ndarray: A 2D array of shape (H, W).
    """
    pred = np.asarray(pred)
    if pred.ndim == 4:      # (B, C, H, W)
        pred = pred[0, 0]
    elif pred.ndim == 3:    # (C, H, W) or (B, H, W)
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Prediction must become 2D, got shape {pred.shape}")
    return pred

def predict_binary_mask(tif_path: str, model, threshold: float = 0.5) -> np.ndarray:
    """
    Predict a binary mask for a given GeoTIFF file using a trained model.

    Args:
        tif_path (str): Path to the input GeoTIFF file.
        model: Trained model used for prediction.
        threshold (float, optional): Threshold for converting probabilities to binary values. Defaults to 0.5.

    Returns:
        np.ndarray: Binary mask of shape (H, W) with values in {0, 1}.
    """
    riverbands, _, _, _, _ = riverpp.open_tif(tif_path)

    pp_riverbands = riverpp.preprocess_bands(riverbands)
    pp_riverbands = riverpp.select_bands(pp_riverbands, [0,1,3,4,5,6,7,8,9]) # Should be added as *args and/or **kwargs

    pred = riverpred.predict_river_extent(pp_riverbands, model)
    pred_hw = squeeze_to_hw(pred)

    return (pred_hw > threshold).astype(np.uint8)


def tile_window_in_mosaic(tile_path: str, mosaic_transform: Affine) -> Window:
    """
    Compute the destination window in a mosaic for a given tile based on its bounds.

    Args:
        tile_path (str): Path to the input tile GeoTIFF file.
        mosaic_transform (Affine): Affine transformation of the mosaic.

    Returns:
        Window: Rasterio Window object representing the destination window in the mosaic.
    """
    with rasterio.open(tile_path) as src:
        win = from_bounds(*src.bounds, transform=mosaic_transform)
    return win.round_offsets().round_lengths()


def match_mask_to_window(mask_hw: np.ndarray, win: Window) -> np.ndarray:
    """
    Crop the binary mask to match the dimensions of the given window if there is an off-by-1 difference due to rounding.

    Args:
        mask_hw (np.ndarray): Binary mask of shape (H, W).
        win (Window): Rasterio Window object with target height and width.

    Returns:
        np.ndarray: Cropped binary mask of shape (win.height, win.width).
    """
    h, w = int(win.height), int(win.width)
    if mask_hw.shape == (h, w):
        return mask_hw
    return mask_hw[:h, :w]


# ----------------------------
# 5) Merge logic (OR into output raster)
# ----------------------------

def initialize_output(dst: rasterio.io.DatasetWriter, fill_value: int = 0) -> None:
    """
    Fill the destination dataset with a constant value.

    Args:
        dst (rasterio.io.DatasetWriter): The destination dataset to be initialized.
        fill_value (int, optional): The constant value to fill the dataset with. Defaults to 0.

    Returns:
        None
    """
    dst.write(np.full((dst.height, dst.width), fill_value, dtype=np.uint8), 1)


def or_write_window(dst: rasterio.io.DatasetWriter, win: Window, mask_hw: np.ndarray) -> None:
    """
    Perform a bitwise OR operation between the existing data in the specified window 
    and the provided binary mask, then write the result back to the destination dataset.

    Args:
        dst (rasterio.io.DatasetWriter): The destination dataset to be updated.
        win (Window): Rasterio Window object specifying the region to update.
        mask_hw (np.ndarray): Binary mask of shape matching the window dimensions.

    Returns:
        None
    """
    existing = dst.read(1, window=win)
    merged = (existing | mask_hw).astype(np.uint8)
    dst.write(merged, 1, window=win)

# ----------------------------
# 6) Orchestrator
# ----------------------------

def mosaic_predictions(tif_paths: List[str], model, out_path: str, threshold: float = 0.5) -> str:
    """
    Create a single large binary (0/1) GeoTIFF by streaming predictions tile-by-tile.

    Args:
        tif_paths (List[str]): List of paths to input GeoTIFF tiles.
        model: Trained model used for predicting binary masks.
        out_path (str): Path to save the output mosaic GeoTIFF.
        threshold (float, optional): Threshold for converting probabilities to binary values. Defaults to 0.5.

    Returns:
        str: Path to the saved output mosaic GeoTIFF.
    """
    spec = build_mosaic_spec(tif_paths)
    meta = build_output_meta(spec)

    with rasterio.open(out_path, "w+", **meta) as dst:
        initialize_output(dst, fill_value=0)

        for p in tif_paths:
            mask = predict_binary_mask(p, model, threshold=threshold)
            win = tile_window_in_mosaic(p, spec.transform)
            mask = match_mask_to_window(mask, win)
            or_write_window(dst, win, mask)

    return out_path




if __name__ == "__main__":
    print("Hello world")
