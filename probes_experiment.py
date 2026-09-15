import json
import math
import random
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# =============================================================================
# 1. CORE DATA STRUCTURES & CONTAINERS
# =============================================================================

@dataclass
class TargetImage:
    """Stores target image representations and metadata."""
    name: str
    image_path: Path
    rgba_image: Optional[np.ndarray] = None  # (H, W, 4) uint8
    label_map: Optional[np.ndarray] = None   # (H, W) int32, 0 is background
    one_hot_map: Optional[np.ndarray] = None # (H, W, N) uint8/bool

    @property
    def shape(self) -> Tuple[int, int]:
        if self.rgba_image is None:
            raise ValueError("RGBA image array is not loaded.")
        return self.rgba_image.shape[0], self.rgba_image.shape[1]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def num_segments(self) -> int:
        if self.label_map is None:
            return 0
        unique_labels = np.unique(self.label_map)
        return int(np.count_nonzero(unique_labels > 0))

    def validate(self) -> bool:
        """Validates internal image representations and arrays."""
        if self.rgba_image is None or self.rgba_image.ndim != 3 or self.rgba_image.shape[2] != 4:
            return False
        if self.label_map is not None:
            if self.label_map.shape[:2] != self.rgba_image.shape[:2]:
                return False
        if self.one_hot_map is not None:
            if self.one_hot_map.shape[:2] != self.rgba_image.shape[:2]:
                return False
        return True

    def save_to_disk(self, export_dir: Union[str, Path]) -> Path:
        """
        Saves all image representations and metadata to a single self-contained directory.

        Directory structure created:
            <export_dir>/<name>/
                ├── rgba.png
                ├── maps.npz        (contains label_map and one_hot_map)
                └── metadata.json
        """
        target_dir = Path(export_dir) / self.name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save RGBA image as PNG
        if self.rgba_image is not None:
            pil_img = Image.fromarray(self.rgba_image).convert("RGBA")
            pil_img.save(target_dir / "rgba.png")

        # 2. Save maps into a compressed NumPy zip archive (.npz)
        maps_to_save = {}
        if self.label_map is not None:
            maps_to_save["label_map"] = self.label_map
        if self.one_hot_map is not None:
            maps_to_save["one_hot_map"] = self.one_hot_map

        if maps_to_save:
            np.savez_compressed(target_dir / "maps.npz", **maps_to_save)

        # 3. Save metadata descriptor
        meta = {
            "name": self.name,
            "has_rgba": self.rgba_image is not None,
            "has_label_map": self.label_map is not None,
            "has_one_hot_map": self.one_hot_map is not None,
            "num_segments": self.num_segments,
            "width": self.width if self.rgba_image is not None else None,
            "height": self.height if self.rgba_image is not None else None,
        }
        with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return target_dir

    @classmethod
    def load_from_disk(cls, target_dir: Union[str, Path]) -> "TargetImage":
        """Loads a complete TargetImage instance from a self-contained bundle directory."""
        bundle_path = Path(target_dir)
        if not bundle_path.exists() or not bundle_path.is_dir():
            raise FileNotFoundError(f"TargetImage bundle directory not found: {bundle_path}")

        meta_path = bundle_path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in bundle: {bundle_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        rgba_path = bundle_path / "rgba.png"
        rgba_arr = None
        if rgba_path.exists():
            pil_img = Image.open(rgba_path).convert("RGBA")
            rgba_arr = np.array(pil_img)

        label_map = None
        one_hot_map = None
        maps_path = bundle_path / "maps.npz"
        if maps_path.exists():
            with np.load(maps_path) as npz:
                if "label_map" in npz:
                    label_map = npz["label_map"]
                if "one_hot_map" in npz:
                    one_hot_map = npz["one_hot_map"]

        return cls(
            name=meta["name"],
            image_path=rgba_path,
            rgba_image=rgba_arr,
            label_map=label_map,
            one_hot_map=one_hot_map,
        )


@dataclass
class Probe:
    """Container for probe points in both pixel and normalized coordinates."""
    points_px: List[Tuple[int, int]]
    points_norm: List[Tuple[float, float]]

    @classmethod
    def from_pixel_points(cls, points_px: List[Tuple[int, int]], img_w: int, img_h: int) -> "Probe":
        """Factory method to build a Probe from absolute pixel coordinates."""
        points_norm = [
            (round(x / float(img_w), 5), round(y / float(img_h), 5))
            for x, y in points_px
        ]
        return cls(points_px=points_px, points_norm=points_norm)

    @classmethod
    def from_normalized_points(cls, points_norm: List[Tuple[float, float]], img_w: int, img_h: int) -> "Probe":
        """Factory method to build a Probe from normalized coordinates [0.0, 1.0]."""
        points_px = [
            (int(round(x * img_w)), int(round(y * img_h)))
            for x, y in points_norm
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
        
    def validate(self) -> bool:
        if not self.points_px or not self.points_norm:
            return False
        if len(self.points_px) != len(self.points_norm):
            return False

        valid_px = all(
            isinstance(pt, tuple) and len(pt) == 2 and isinstance(pt[0], int) and isinstance(pt[1], int)
            for pt in self.points_px
        )
        valid_norm = all(
            isinstance(pt, tuple) and len(pt) == 2 and 0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0
            for pt in self.points_norm
        )
        return valid_px and valid_norm


@dataclass
class Trial:
    """Represents a single experimental trial."""
    trial_id: str
    task_mode: str  # "vision" or "imagery"
    target_image: TargetImage
    probe: Optional[Probe] = None
    probe_result: Optional[Dict[str, Any]] = None
    rendered_stimulus_path: Optional[str] = None

    def validate(self) -> bool:
        if self.task_mode not in ("vision", "imagery"):
            return False
        if not self.target_image or not self.target_image.validate():
            return False
        if self.probe is not None and not self.probe.validate():
            return False
        return True


@dataclass
class Run:
    """Group of trials associated with a single TargetImage."""
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

@dataclass
class RunConfig:
    """Blueprint for specifying trial structures per target image."""
    target_image: TargetImage
    num_vision_trials: int
    num_imagery_trials: int
    vision_repeats: int = 1
    imagery_repeats: int = 1


@dataclass
class Experiment:
    """Top-level container managing runs, stimulus styles, and data export."""
    experiment_id: str
    runs: List[Run] = field(default_factory=list)
    root_dir: Path = field(default_factory=lambda: Path("./experiment_output"))

    # Probe Rendering Style Specs
    probe_color: Tuple[int, int, int, int] = (255, 0, 0, 255)
    probe_radius: int = 5
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 255)

    @classmethod
    def create(
        cls,
        experiment_id: str,
        run_configs: List[RunConfig],
        root_dir: Union[str, Path] = "./experiment_output",
        probe_color: Tuple[int, int, int, int] = (255, 0, 0, 255),
        probe_radius: int = 5,
        bg_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
    ) -> "Experiment":
        exp = cls(
            experiment_id=experiment_id,
            root_dir=Path(root_dir),
            probe_color=probe_color,
            probe_radius=probe_radius,
            bg_color=bg_color,
        )
        exp.root_dir.mkdir(parents=True, exist_ok=True)

        for config in run_configs:
            run = Run(run_id=f"run_{config.target_image.name}", target_image=config.target_image)

            # Populate vision trials
            for i in range(config.num_vision_trials):
                for r in range(config.vision_repeats):
                    run.trials.append(
                        Trial(
                            trial_id=f"{run.run_id}_vis_p{i:03d}_r{r}",
                            task_mode="vision",
                            target_image=config.target_image,
                        )
                    )

            # Populate imagery trials
            for j in range(config.num_imagery_trials):
                for r in range(config.imagery_repeats):
                    run.trials.append(
                        Trial(
                            trial_id=f"{run.run_id}_img_p{j:03d}_r{r}",
                            task_mode="imagery",
                            target_image=config.target_image,
                        )
                    )

            exp.runs.append(run)
        return exp

    def validate(self) -> bool:
        if not self.runs:
            return False
        return all(r.validate() for r in self.runs)

    def to_dataframe(self) -> pd.DataFrame:
        """Flattens experiment structure into a Pandas DataFrame."""
        if not self.validate():
            raise ValueError("Experiment validation failed prior to DataFrame export.")

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
                    "rendered_stimulus_path": trial.rendered_stimulus_path or "",
                    "probe_points_px": str(trial.probe.points_px) if trial.probe else "",
                    "probe_points_norm": str(trial.probe.points_norm) if trial.probe else "",
                }
                # Flatten probe_result keys into distinct columns
                if trial.probe_result:
                    for k, v in trial.probe_result.items():
                        record[f"ground_truth_{k}"] = str(v) if isinstance(v, (list, dict, np.ndarray)) else v

                records.append(record)

        return pd.DataFrame(records)

    def export_summary_txt(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Generates a text summary of the experiment setup."""
        if not self.validate():
            raise ValueError("Experiment validation failed prior to summary export.")

        path = Path(output_path) if output_path else self.root_dir / "experiment_summary.txt"

        lines = [
            f"=== EXPERIMENT SUMMARY: {self.experiment_id} ===",
            f"Total Runs: {len(self.runs)}",
            f"Output Root Directory: {self.root_dir.resolve().as_posix()}",
            f"Probe Style Specs: Radius={self.probe_radius}px | Color={self.probe_color} | BG={self.bg_color}",
            "------------------------------------------------",
        ]

        total_trials = 0
        for run in self.runs:
            vis_count = sum(1 for t in run.trials if t.task_mode == "vision")
            img_count = sum(1 for t in run.trials if t.task_mode == "imagery")
            total_trials += len(run.trials)

            lines.append(f"Run ID: {run.run_id}")
            lines.append(f"  Target Image: {run.target_image.name} ({run.target_image.width}x{run.target_image.height})")
            lines.append(f"  Segments Count: {run.target_image.num_segments}")
            lines.append(f"  Vision Trials: {vis_count} | Imagery Trials: {img_count}")
            lines.append("------------------------------------------------")

        lines.append(f"Total Trials Across Experiment: {total_trials}")
        summary_text = "\n".join(lines)

        path.write_text(summary_text)
        return summary_text

    def export_target_assets(self) -> List[Path]:
        """Exports all unique target image bundles to the experiment root directory."""
        assets_dir = self.root_dir / "target_assets"
        saved_paths = []

        seen_names = set()
        for run in self.runs:
            if run.target_image.name not in seen_names:
                bundle_path = run.target_image.save_to_disk(assets_dir)
                saved_paths.append(bundle_path)
                seen_names.add(run.target_image.name)

        return saved_paths

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
                "rgba_image": img.rgba_image,
                "label_map": img.label_map,
                "one_hot_map": img.one_hot_map,
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
                first_results = cond_trials[0].probe_result
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

# =============================================================================
# 2. TARGET SEGMENTATION (K-MEANS)
# =============================================================================

class TargetSegmenter:
    """K-Means segmentation processor for RGB/RGBA TargetImages."""

    @staticmethod
    def segment_kmeans(target_image: TargetImage, n_clusters: int, seed: int = 42) -> None:
        """Performs K-Means color segmentation and updates target_image attributes."""
        if target_image.rgba_image is None:
            raise ValueError("RGBA image array missing in TargetImage.")

        rgb_pixels = target_image.rgba_image[:, :, :3].reshape(-1, 3).astype(np.float32)

        np.random.seed(seed)
        initial_indices = np.random.choice(len(rgb_pixels), size=n_clusters, replace=False)
        centroids = rgb_pixels[initial_indices]

        for _ in range(15):
            distances = np.linalg.norm(rgb_pixels[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                rgb_pixels[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
                for k in range(n_clusters)
            ])
            if np.allclose(centroids, new_centroids, atol=1e-2):
                break
            centroids = new_centroids

        # Map labels to 1..N (reserve 0 for background)
        h, w = target_image.shape
        label_map = (labels.reshape((h, w)) + 1).astype(np.int32)

        # Build one-hot spatial mask tensor (H, W, N)
        one_hot_map = np.zeros((h, w, n_clusters), dtype=np.uint8)
        for k in range(n_clusters):
            one_hot_map[:, :, k] = (label_map == (k + 1)).astype(np.uint8)

        target_image.label_map = label_map
        target_image.one_hot_map = one_hot_map


# =============================================================================
# 3. PROBE GENERATION & SUBSET SAMPLING
# =============================================================================

class ProbeSubsetPolicy(Enum):
    SHARED_EQUAL = auto()
    VISION_SUBSET_OF_IMAGERY = auto()
    IMAGERY_SUBSET_OF_VISION = auto()
    INDEPENDENT = auto()


class GridProbeGenerator:
    """Modular sampling generator prioritizing full grid coverage."""

    def __init__(self, grid_shape: Tuple[int, int], points_per_probe: int = 4):
        self.grid_rows, self.grid_cols = grid_shape
        self.points_per_probe = points_per_probe

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
            # Pre-populate binary mask
            probe.generate_binary_mask(target_image.height, target_image.width)
            probes.append(probe)

        return probes

    def assign_probes_to_run(
        self,
        run: Run,
        policy: ProbeSubsetPolicy = ProbeSubsetPolicy.VISION_SUBSET_OF_IMAGERY,
        seed: Optional[int] = None,
    ) -> None:
        """Applies probes and saves full source grid coordinates to the Run."""
        # Save full grid definition at Run level
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


# =============================================================================
# 4. MASK ANALYSIS & METRIC COMPUTATIONS
# =============================================================================

class ProbeMaskAnalyzer:
    """Vectorized probe statistics evaluator using standard NumPy array indexing."""

    @staticmethod
    def _extract_labels(probe: Probe, label_map: np.ndarray) -> np.ndarray:
        h, w = label_map.shape[:2]
        x_coords = np.clip([p[0] for p in probe.points_px], 0, w - 1)
        y_coords = np.clip([p[1] for p in probe.points_px], 0, h - 1)
        return label_map[y_coords, x_coords]

    @classmethod
    def object_count(cls, probe: Probe, label_map: np.ndarray) -> int:
        labels = cls._extract_labels(probe, label_map)
        return int(np.count_nonzero(np.unique(labels[labels > 0])))

    @classmethod
    def touches_per_segment(cls, probe: Probe, label_map: np.ndarray, num_segments: int) -> Dict[int, int]:
        labels = cls._extract_labels(probe, label_map)
        counts = np.bincount(labels[labels > 0], minlength=num_segments + 1)
        return {seg_id: int(counts[seg_id]) for seg_id in range(1, num_segments + 1)}

    @classmethod
    def target_object_hits(cls, probe: Probe, label_map: np.ndarray, target_segment_id: int) -> int:
        labels = cls._extract_labels(probe, label_map)
        return int(np.sum(labels == target_segment_id))

    @classmethod
    def evaluate_experiment_results(
        cls,
        experiment: Experiment,
        metrics: List[str],
        target_segment_id: Optional[int] = None,
    ) -> None:
        """Computes requested metrics and updates trial.probe_result dictionaries."""
        for run in experiment.runs:
            label_map = run.target_image.label_map
            if label_map is None:
                raise ValueError(f"Label map not initialized for image '{run.target_image.name}'.")

            num_segs = run.target_image.num_segments

            for trial in run.trials:
                if trial.probe is None:
                    continue

                res = {}
                if "object_count" in metrics:
                    res["object_count"] = cls.object_count(trial.probe, label_map)
                if "touches_per_segment" in metrics:
                    res["touches_per_segment"] = cls.touches_per_segment(trial.probe, label_map, num_segs)
                if "target_object_hits" in metrics:
                    if target_segment_id is None:
                        raise ValueError("target_segment_id must be provided for 'target_object_hits' metric.")
                    res[f"target_obj_{target_segment_id}_hits"] = cls.target_object_hits(
                        trial.probe, label_map, target_segment_id
                    )

                trial.probe_result = res


# =============================================================================
# 5. STIMULUS RENDERING
# =============================================================================

class StimulusRenderer:
    """Renders vision and imagery trial stimulus images using PIL and Experiment style specs."""

    @staticmethod
    def render_experiment_stimuli(
        experiment: Experiment,
        include_ground_truth: bool = False,
    ) -> None:
        stimuli_dir = experiment.root_dir / "rendered_stimuli"
        stimuli_dir.mkdir(parents=True, exist_ok=True)

        for run in experiment.runs:
            for trial in run.trials:
                if trial.probe is None:
                    continue

                w, h = run.target_image.width, run.target_image.height

                # Build base canvas using Experiment probe styles
                if trial.task_mode == "vision":
                    canvas = Image.fromarray(run.target_image.rgba_image).convert("RGBA")
                else:
                    canvas = Image.new("RGBA", (w, h), experiment.bg_color)

                draw = ImageDraw.Draw(canvas, mode="RGBA")
                r = experiment.probe_radius
                for x, y in trial.probe.points_px:
                    draw.ellipse(
                        [x - r, y - r, x + r, y + r],
                        fill=experiment.probe_color,
                        outline=(255, 255, 255, 255),
                    )

                # Optional in-place ground truth banner overlay
                if include_ground_truth:
                    banner_height = 28
                    draw.rectangle([(0, 0), (w, banner_height)], fill=(0, 0, 0, 200))
                    results = getattr(trial, "probe_result", "N/A")
                    obj_cnt = results["object_count"] if isinstance(results, dict) else "N/A"
                    touches = results["touches_per_segment"] if isinstance(results, dict) else "N/A"
                    t1_hits = results["target_obj_1_hits"] if isinstance(results, dict) else "N/A"

                    banner_text = f"Objs Hit: {obj_cnt} | Touches/Seg: {touches} | Obj#1 Hits: {t1_hits}"
                    draw.text((8, 6), banner_text, fill=(255, 255, 255, 255))

                out_file = stimuli_dir / f"{trial.trial_id}.png"
                canvas.save(out_file)

                # Store relative POSIX path string
                rel_path = out_file.relative_to(experiment.root_dir).as_posix()
                trial.rendered_stimulus_path = rel_path
