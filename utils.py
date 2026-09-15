from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
from PIL import Image
import re
from probes_experiment import TargetImage


def load_target_images_from_directory(
    directory_path: Union[str, Path],
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    recursive: bool = False
) -> List[TargetImage]:
    """
    Scans a directory for image files and instantiates a TargetImage object for each.

    Args:
        directory_path: Path to the target directory containing images.
        extensions: Tuple of file extensions (case-insensitive) to include.
        recursive: If True, recursively searches subdirectories.

    Returns:
        A list of initialized TargetImage instances.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path.resolve()}")

    pattern = "**/*" if recursive else "*"
    valid_exts = {ext.lower() for ext in extensions}
    
    image_files = [
        p for p in dir_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    target_images = []
    for img_path in sorted(image_files):
        try:
            # Load RGBA image via PIL and convert to NumPy array
            pil_img = Image.open(img_path).convert("RGBA")
            rgba_arr = np.array(pil_img)

            # Instantiate TargetImage directly via dataclass constructor
            target_img = TargetImage(
                name=img_path.stem,
                image_path=img_path,
                rgba_image=rgba_arr
            )
            target_images.append(target_img)
        except Exception as e:
            print(f"Warning: Failed to load '{img_path.name}': {e}")

    return target_images


def extract_segment_count_from_filename(
    image_path: Union[str, Path], 
    default: Optional[int] = None
) -> Optional[int]:
    """
    Extracts the segment count encoded at the end of an image filename.

    Examples:
        "scene_04.png"          -> 4
        "stimulus_seg_12.jpg"   -> 12
        "target_image3.png"     -> 3

    Args:
        image_path: Path or filename of the target image.
        default: Fallback integer to return if no trailing digits are found.

    Returns:
        The extracted segment count as an integer, or the default value.
    """
    stem = Path(image_path).stem  # Removes directory path and extension
    
    # Matches one or more digits at the end of the string
    match = re.search(r'(\d+)$', stem)
    
    if match:
        return int(match.group(1))
    
    return default

from typing import List, Tuple, Union
import numpy as np


def sample_grid_map(
    target_map: np.ndarray, 
    grid_points: Union[List[Tuple[int, int]], np.ndarray]
) -> np.ndarray:
    """
    Downsamples a spatial map at a set of regular grid points.
    
    Parameters
    ----------
    target_map : np.ndarray
        Array of shape (H, W) for label maps or (H, W, C) for one-hot maps.
    grid_points : List[Tuple[int, int]] or np.ndarray
        Coordinates as (x, y) pixel integers representing grid cell centers.
        
    Returns
    -------
    sampled_map : np.ndarray
        Array of shape (rows, cols) for label_maps, or (rows, cols, C) for one-hot maps,
        matching the spatial dimensions of the target grid.
    """
    pts = np.asarray(grid_points, dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("grid_points must be an (N, 2) sequence of (x, y) coordinates.")
    
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)

    # 1. Check completeness: points must fill a complete unique_y x unique_x product
    expected_count = len(unique_y) * len(unique_x)
    if len(pts) != expected_count:
        raise ValueError(
            f"Incomplete grid: Expected {expected_count} unique points "
            f"({len(unique_y)} rows x {len(unique_x)} cols), but got {len(pts)}."
        )

    # 2. Check regularity: steps between consecutive unique coordinates must be equal
    if len(unique_x) > 1:
        dx = np.diff(unique_x)
        if not np.all(dx == dx[0]):
            raise ValueError(f"Irregular grid spacing along X axis: step sizes = {dx}")
            
    if len(unique_y) > 1:
        dy = np.diff(unique_y)
        if not np.all(dy == dy[0]):
            raise ValueError(f"Irregular grid spacing along Y axis: step sizes = {dy}")

    # 3. Verify all Cartesian combinations exist in input
    grid_set = set(map(tuple, pts))
    expected_set = {(x, y) for y in unique_y for x in unique_x}
    if grid_set != expected_set:
        raise ValueError("Irregular grid layout: Points do not form a complete, axis-aligned rectangular grid.")

    # 4. Perform grid extraction (np.ix_ preserves row/col grid dimensions)
    sampled = target_map[np.ix_(unique_y, unique_x)]
    return sampled

from copy import deepcopy
from typing import Dict
import numpy as np


def downsample_data_bundle(vec_dict: Dict, inplace: bool = False) -> Dict:
    """
    Downsamples all target assets and probe masks in a vectorized data bundle
    to match the sampling grid resolution.

    - Applies `sample_grid_map` to label_map, rgb, rgba, one_hot, and probe masks.
    - Removes all 'points_px' references.
    - Asserts downsampled probe mask pixel counts match points per probe.

    Parameters
    ----------
    vec_dict : Dict
        Vectorized data bundle output from `Experiment.to_data_bundle()`.
    inplace : bool, optional
        If True, mutates vec_dict directly. Otherwise returns a modified copy.

    Returns
    -------
    Dict
        The downsampled vectorized data bundle.
    """
    bundle = vec_dict if inplace else deepcopy(vec_dict)

    for image_name, data in bundle.items():
        sampling_grid = data["sampling_grid"].get("points_px")
        if sampling_grid is None or len(sampling_grid) == 0:
            continue

        # ---------------------------------------------------------------------
        # 1. Downsample Target Image Assets & Update Metadata
        # ---------------------------------------------------------------------
        assets = data["target_assets"]

        for map_key in ["label_map", "rgb", "rgba", "one_hot"]:
            if assets.get(map_key) is not None:
                assets[map_key] = sample_grid_map(assets[map_key], sampling_grid)

        # Update height and width dimensions to match downsampled shape
        downsampled_h, downsampled_w = assets["label_map"].shape[:2]
        assets["dimensions"] = {"height": downsampled_h, "width": downsampled_w}

        # ---------------------------------------------------------------------
        # 2. Downsample Condition Probe Masks & Purge Pixel Points
        # ---------------------------------------------------------------------
        for condition in ["vision", "imagery"]:
            cond_data = data.get(condition)
            if not cond_data or cond_data.get("num_trials", 0) == 0:
                continue

            probes = cond_data["probes"]

            # Remove absolute pixel points references
            probes.pop("points_px", None)

            # Downsample and validate binary masks
            if "masks" in probes and probes["masks"].size > 0:
                orig_masks = probes["masks"]  # Shape: (T, H, W)
                downsampled_masks = []

                # Expected active pixels per probe (points per probe count)
                if "points_norm" in probes and probes["points_norm"].size > 0:
                    expected_on_pixels = probes["points_norm"].shape[1]
                else:
                    expected_on_pixels = None

                for trial_idx, mask in enumerate(orig_masks):
                    ds_mask = sample_grid_map(mask, sampling_grid)
                    actual_on_pixels = int(np.sum(ds_mask))

                    # Verify 'on' pixel count
                    if expected_on_pixels is not None and actual_on_pixels != expected_on_pixels:
                        raise ValueError(
                            f"Mask downsampling mismatch in {image_name} [{condition}] trial index {trial_idx}: "
                            f"Expected {expected_on_pixels} 'on' pixels, but found {actual_on_pixels}."
                        )

                    downsampled_masks.append(ds_mask)

                probes["masks"] = np.stack(downsampled_masks, axis=0)

        # ---------------------------------------------------------------------
        # 3. Purge Sampling Grid Pixel Points
        # ---------------------------------------------------------------------
        data["sampling_grid"].pop("points_px", None)

    return bundle
