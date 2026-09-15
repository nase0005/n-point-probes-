"""
probes_experiment.py

A modular framework for generating spatial probe grids, running vision/imagery
trials across target images, and exporting vectorized data bundles for analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import numpy as np
import pandas as pd
from PIL import Image


class ProbeSubsetPolicy(Enum):
    """Defines how probe sets are shared or sampled across tasks."""
    VISION_SUBSET_OF_IMAGERY = "vision_subset_of_imagery"
    IMAGERY_SUBSET_OF_VISION = "imagery_subset_of_vision"
    SHARED_EQUAL = "shared_equal"
    INDEPENDENT = "independent"


@dataclass
class TargetImage:
    """Holds metadata and pixel arrays for a target image stimulus."""
    name: str
    image_path: Path
    rgb: Optional[np.ndarray] = field(default=None, repr=False)
    rgba: Optional[np.ndarray] = field(default=None, repr=False)
    label_map: Optional[np.ndarray] = field(default=None, repr=False)
    one_hot: Optional[np.ndarray] = field(default=None, repr=False)
    num_segments: int = 0

    @property
    def height(self) -> int:
        if self.rgb is not None:
            return self.rgb.shape[0]
        return 0

    @property
    def width(self) -> int:
        if self.rgb is not None:
            return self.rgb.shape[1]
        return 0

    def validate(self) -> bool:
        return self.rgb is not None and self.height > 0 and self.width > 0


@dataclass
class Probe:
    """Container for probe points and associated spatial binary masks."""
    points_px: List[Tuple[int, int]]
    points_norm: List[Tuple[float, float]]
    binary_mask: Optional[np.ndarray] = field(default=None, repr=False)  # Shape: (H, W), bool

    @classmethod
    def from_pixel_points(
        cls, 
        points_px: List[Tuple[int, int]], 
        img_w: int, 
        img_h: int
    ) -> "Probe":
        """Factory method to build a Probe from absolute pixel coordinates."""
        points_norm = [
            (round(x / float(img_w), 5), round(y / float(img_h), 5))
            for x, y in points_px
        ]
        return cls(points_px=points_px, points_norm=points_norm)

    def generate_binary_mask(self, height: int, width: int, radius: int = 0) -> np.ndarray:
        """Generates and stores a 2D boolean mask array (Height, Width)."""
        mask = np.zeros((height, width), dtype=bool)
        
        if radius <= 0:
            for x, y in self.points_px:
                x_clamped = np.clip(x, 0, width - 1)
                y_clamped = np.clip(y, 0, height - 1)
                mask[y_clamped, x_clamped] = True
        else:
            y_grid, x_grid = np.ogrid[:height, :width]
            for x, y in self.points_px:
                x_clamped = np.clip(x, 0, width - 1)
                y_clamped = np.clip(y, 0, height - 1)
                dist_sq = (x_grid - x_clamped) ** 2 + (y_grid - y_clamped) ** 2
                mask |= (dist_sq <= radius ** 2)

        self.binary_mask = mask
        return mask


@dataclass
class Trial:
    """Represents a single experimental trial."""
    trial_id: str
    task_mode: str  # "vision" or "imagery"
    target_image: TargetImage
    rendered_stimulus_path: Optional[str] = None
    probe: Optional[Probe] = None
    results: Dict[str, Union[int, float, Dict, np.ndarray]] = field(default_factory=dict)

    def validate(self) -> bool:
        return bool(self.trial_id and self.task_mode in ("vision", "imagery"))


@dataclass
class Run:
    """Group of trials associated with a single TargetImage and shared sampling grid."""
    run_id: str
    target_image: TargetImage
    trials: List[Trial] = field(default_factory=list)
    grid_points_px: Optional[List[Tuple[int, int]]] = None
    grid_points_norm: Optional[List[Tuple[float, float]]] = None

    def validate(self) -> bool:
        if not self.trials:
            return False
        if not self.target_image or not self.target_image.validate():
            return False
        return all(t.validate() for t in self.trials)


class GridProbeGenerator:
    """Modular sampling generator prioritizing full grid coverage."""

    def __init__(self, grid_shape: Tuple[int, int], points_per_probe: int = 4, probe_radius: int = 0):
        self.grid_rows, self.grid_cols = grid_shape
        self.points_per_probe = points_per_probe
        self.probe_radius = probe_radius

    def _get_grid_centers(self, img_w: int, img_h: int) -> List[Tuple[int, int]]:
        x_edges = np.linspace(0, img_w, self.grid_cols + 1)
        y_edges = np.linspace(0, img_h, self.grid_rows + 1)

        centers = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cx = int((x_edges[c] + x_edges[c + 1]) / 2)
                cy = int((y_edges[r] + y_edges[r + 1]) / 2)
                centers.append((cx, cy))
        return centers

    def generate_probes(self, target_image: TargetImage, count: int, seed: Optional[int] = None) -> List[Probe]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        grid_centers = self._get_grid_centers(target_image.width, target_image.height)
        num_cells = len(grid_centers)
        total_points_needed = count * self.points_per_probe

        sampled_points = []
        if total_points_needed >= num_cells:
            shuffled_centers = list(grid_centers)
            random.shuffle(shuffled_centers)
            sampled_points.extend(shuffled_centers)

            remaining = total_points_needed - num_cells
            if remaining > 0:
                sampled_points.extend(random.choices(grid_centers, k=remaining))
        else:
            sampled_points = random.sample(grid_centers, k=total_points_needed)

        random.shuffle(sampled_points)

        probes = []
        for i in range(0, len(sampled_points), self.points_per_probe):
            chunk_px = sampled_points[i : i + self.points_per_probe]
            probe = Probe.from_pixel_points(chunk_px, target_image.width, target_image.height)
            probe.generate_binary_mask(target_image.height, target_image.width, radius=self.probe_radius)
            probes.append(probe)

        return probes

    def assign_probes_to_run(
        self,
        run: Run,
        policy: ProbeSubsetPolicy = ProbeSubsetPolicy.VISION_SUBSET_OF_IMAGERY,
        seed: Optional[int] = None,
    ) -> None:
        """Applies probes and stores run-level sampling grid coordinates."""
        grid_px = self._get_grid_centers(run.target_image.width, run.target_image.height)
        grid_norm = [
            (round(x / float(run.target_image.width), 5), round(y / float(run.target_image.height), 5))
            for x, y in grid_px
        ]
        run.grid_points_px = grid_px
        run.grid_points_norm = grid_norm

        vis_trials = [t for t in run.trials if t.task_mode == "vision"]
        img_trials = [t for t in run.trials if t.task_mode == "imagery"]

        n_vis_unique = len(set(t.trial_id.rsplit("_r", 1)[0] for t in vis_trials))
        n_img_unique = len(set(t.trial_id.rsplit("_r", 1)[0] for t in img_trials))

        max_unique = max(n_vis_unique, n_img_unique)
        master_pool = self.generate_probes(run.target_image, count=max_unique, seed=seed)

        if policy == ProbeSubsetPolicy.VISION_SUBSET_OF_IMAGERY:
            img_probes = master_pool[:n_img_unique]
            vis_probes = random.sample(master_pool, k=n_vis_unique)
        elif policy == ProbeSubsetPolicy.IMAGERY_SUBSET_OF_VISION:
            vis_probes = master_pool[:n_vis_unique]
            img_probes = random.sample(master_pool, k=n_img_unique)
        elif policy == ProbeSubsetPolicy.SHARED_EQUAL:
            vis_probes = master_pool[:n_vis_unique]
            img_probes = master_pool[:n_img_unique]
        elif policy == ProbeSubsetPolicy.INDEPENDENT:
            vis_probes = self.generate_probes(run.target_image, count=n_vis_unique, seed=seed)
            img_probes = self.generate_probes(run.target_image, count=n_img_unique, seed=seed)

        def _attach(trials: List[Trial], probe_pool: List[Probe]):
            trial_groups: Dict[str, List[Trial]] = {}
            for t in trials:
                base_id = t.trial_id.rsplit("_r", 1)[0]
                trial_groups.setdefault(base_id, []).append(t)
            for (base_id, t_list), prb in zip(trial_groups.items(), probe_pool):
                for t in t_list:
                    t.probe = prb

        _attach(vis_trials, vis_probes)
        _attach(img_trials, img_probes)


@dataclass
class Experiment:
    """Top-level manager holding multiple runs and exporting formats."""
    experiment_id: str
    runs: List[Run] = field(default_factory=list)

    def validate(self) -> bool:
        if not self.runs:
            return False
        return all(r.validate() for r in self.runs)

    def to_dataframe(self) -> pd.DataFrame:
        """Exports trial metadata to a flattened pandas DataFrame."""
        records = []
        for run in self.runs:
            for trial in run.trials:
                record = {
                    "experiment_id": self.experiment_id,
                    "run_id": run.run_id,
                    "trial_id": trial.trial_id,
                    "task_mode": trial.task_mode,
                    "target_image_name": trial.target_image.name,
                    "target_image_path": str(trial.target_image.image_path.as_posix()),
                    "target_image_num_segments": trial.target_image.num_segments,
                    "run_grid_points_px": str(run.grid_points_px) if run.grid_points_px else "",
                    "run_grid_points_norm": str(run.grid_points_norm) if run.grid_points_norm else "",
                    "rendered_stimulus_path": trial.rendered_stimulus_path or "",
                    "probe_points_px": str(trial.probe.points_px) if trial.probe else "",
                    "probe_points_norm": str(trial.probe.points_norm) if trial.probe else "",
                }
                # Add scalar result metrics
                for k, v in trial.results.items():
                    if isinstance(v, (int, float, str, bool)):
                        record[f"result_{k}"] = v
                records.append(record)
        return pd.DataFrame(records)

    def to_data_bundle(self) -> Dict[str, Dict]:
        """
        Exports a nested dictionary data bundle organized by target image.
        
        Trial data for vision and imagery conditions are stacked into concatenated 
        NumPy arrays along a leading trial dimension.
        """
        bundle = {}

        for run in self.runs:
            img = run.target_image
            target_key = img.name

            # Target assets sub-dictionary
            target_assets = {
                "name": img.name,
                "image_path": str(img.image_path.as_posix()),
                "dimensions": {"height": img.height, "width": img.width},
                "num_segments": img.num_segments,
                "rgb": img.rgb,
                "rgba": img.rgba,
                "label_map": img.label_map,
                "one_hot": img.one_hot,
            }

            # Sampling grid sub-dictionary
            sampling_grid = {
                "points_px": np.array(run.grid_points_px, dtype=np.int32) if run.grid_points_px else np.array([]),
                "points_norm": np.array(run.grid_points_norm, dtype=np.float32) if run.grid_points_norm else np.array([]),
            }

            # Helper function to vectorize condition trials
            def _build_condition_bundle(task_mode: str) -> Dict:
                cond_trials = [t for t in run.trials if t.task_mode == task_mode]
                n_trials = len(cond_trials)

                if n_trials == 0:
                    return {
                        "num_trials": 0,
                        "trial_ids": np.array([], dtype=str),
                        "probes": {"points_px": np.array([]), "points_norm": np.array([]), "masks": np.array([])},
                        "results": {},
                    }

                trial_ids = np.array([t.trial_id for t in cond_trials], dtype=object)

                # Concatenate probe spatial coordinates and masks
                pts_px_list = [t.probe.points_px for t in cond_trials if t.probe]
                pts_norm_list = [t.probe.points_norm for t in cond_trials if t.probe]
                masks_list = [
                    t.probe.binary_mask 
                    if (t.probe and t.probe.binary_mask is not None) 
                    else np.zeros((img.height, img.width), dtype=bool) 
                    for t in cond_trials
                ]

                probes_data = {
                    "points_px": np.array(pts_px_list, dtype=np.int32),
                    "points_norm": np.array(pts_norm_list, dtype=np.float32),
                    "masks": np.stack(masks_list, axis=0) if masks_list else np.array([]),
                }

                # Concatenate scalar and dictionary/array results
                results_data = {}
                first_results = cond_trials[0].results
                for res_key, res_val in first_results.items():
                    res_type_list = [t.results.get(res_key) for t in cond_trials]
                    if isinstance(res_val, (int, float, bool)):
                        results_data[res_key] = np.array(res_type_list)
                    elif isinstance(res_val, (list, np.ndarray)):
                        results_data[res_key] = np.stack(res_type_list, axis=0)
                    elif isinstance(res_val, dict):
                        # Convert dict metrics (e.g., touches_per_segment) into 2D stacked array
                        dict_keys = list(res_val.keys())
                        matrix = [[t.results[res_key].get(dk, 0) for dk in dict_keys] for t in cond_trials]
                        results_data[res_key] = np.array(matrix)

                return {
                    "num_trials": n_trials,
                    "trial_ids": trial_ids,
                    "probes": probes_data,
                    "results": results_data,
                }

            bundle[target_key] = {
                "target_assets": target_assets,
                "sampling_grid": sampling_grid,
                "vision": _build_condition_bundle("vision"),
                "imagery": _build_condition_bundle("imagery"),
            }

        return bundle