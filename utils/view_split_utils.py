import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.read_write_model import qvec2rotmat, read_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ViewRecord:
    global_index: int
    colmap_image_id: int
    camera_id: int
    image_name: str
    image_path: str
    qvec: list
    tvec: list
    camera_center: list
    width: int
    height: int
    image_file_exists: bool
    split: str = ""
    train_pool_position: Optional[int] = None


def natural_lower_key(name: str):
    base = str(name).replace("\\", "/").split("/")[-1].lower()
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", base)]


def normalized_rel_name(name: str) -> str:
    return str(name).replace("\\", "/").strip().lower()


def basename_key(name: str) -> str:
    return os.path.basename(str(name).replace("\\", "/")).lower()


def camera_center_from_image(image) -> np.ndarray:
    rot = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
    tvec = np.asarray(image.tvec, dtype=np.float64)
    return -rot.T @ tvec


def camera_view_direction_from_image(image) -> np.ndarray:
    rot = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
    direction = rot.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 0:
        return direction
    return direction / norm


def _list_image_files(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images directory not found: {images_dir}")

    paths = []
    for root, _, files in os.walk(images_dir):
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, filename))
    return sorted(paths, key=lambda p: natural_lower_key(os.path.relpath(p, images_dir)))


def _build_image_lookup(image_files: Sequence[str], images_dir: str):
    by_rel: Dict[str, str] = {}
    by_base: Dict[str, List[str]] = {}
    by_stem: Dict[str, List[str]] = {}
    for path in image_files:
        rel = normalized_rel_name(os.path.relpath(path, images_dir))
        base = basename_key(path)
        stem = Path(base).stem.lower()
        by_rel[rel] = path
        by_base.setdefault(base, []).append(path)
        by_stem.setdefault(stem, []).append(path)
    duplicate_file_stems = sorted(
        [stem for stem, paths in by_stem.items() if len(paths) > 1],
        key=natural_lower_key,
    )
    return by_rel, by_base, by_stem, duplicate_file_stems


def load_full_colmap_model(source_path: str):
    sparse_dir = os.path.join(source_path, "sparse", "0")
    missing = [
        name for name in ("cameras.bin", "images.bin", "points3D.bin")
        if not os.path.isfile(os.path.join(sparse_dir, name))
    ]
    if missing:
        raise FileNotFoundError(f"missing COLMAP files under {sparse_dir}: {missing}")
    return read_model(sparse_dir, ext=".bin")


def load_full_colmap_views(
    source_path: str,
    images_dir_name: str = "images",
    return_model: bool = False,
):
    source_path = os.path.abspath(source_path)
    images_dir = os.path.join(source_path, images_dir_name)
    cameras, images, points3D = load_full_colmap_model(source_path)
    image_files = _list_image_files(images_dir)
    by_rel, by_base, by_stem, duplicate_file_stems = _build_image_lookup(image_files, images_dir)

    missing_image_names = []
    ambiguous_image_names = []
    records = []

    sorted_images = sorted(images.values(), key=lambda img: natural_lower_key(img.name))
    for image in sorted_images:
        if int(image.camera_id) not in cameras:
            raise ValueError(
                f"COLMAP image {image.name!r} references missing camera_id={image.camera_id}"
            )

        rel_key = normalized_rel_name(image.name)
        base_key = basename_key(image.name)
        stem_key = Path(base_key).stem.lower()

        image_path = by_rel.get(rel_key)
        if image_path is None:
            base_matches = by_base.get(base_key, [])
            if len(base_matches) == 1:
                image_path = base_matches[0]
            elif len(base_matches) > 1:
                ambiguous_image_names.append(image.name)

        if image_path is None:
            stem_matches = by_stem.get(stem_key, [])
            if len(stem_matches) == 1:
                image_path = stem_matches[0]
            elif len(stem_matches) > 1:
                ambiguous_image_names.append(image.name)

        exists = bool(image_path and os.path.isfile(image_path))
        if not exists:
            missing_image_names.append(image.name)
            image_path = os.path.join(images_dir, *str(image.name).replace("\\", "/").split("/"))

        camera = cameras[int(image.camera_id)]
        records.append(
            ViewRecord(
                global_index=-1,
                colmap_image_id=int(image.id),
                camera_id=int(image.camera_id),
                image_name=image.name,
                image_path=os.path.abspath(image_path),
                qvec=np.asarray(image.qvec, dtype=np.float64).tolist(),
                tvec=np.asarray(image.tvec, dtype=np.float64).tolist(),
                camera_center=camera_center_from_image(image).tolist(),
                width=int(camera.width),
                height=int(camera.height),
                image_file_exists=exists,
            )
        )

    records = sorted(records, key=lambda v: natural_lower_key(v.image_name))
    for idx, record in enumerate(records):
        record.global_index = idx

    audit = {
        "source_path": source_path,
        "images_dir": os.path.abspath(images_dir),
        "full_view_count": len(records),
        "full_colmap_image_count": len(images),
        "full_image_file_count": len(image_files),
        "missing_image_file_count": len(missing_image_names),
        "missing_image_names": sorted(set(missing_image_names), key=natural_lower_key),
        "ambiguous_image_name_count": len(set(ambiguous_image_names)),
        "ambiguous_image_names": sorted(set(ambiguous_image_names), key=natural_lower_key),
        "duplicate_file_stems": duplicate_file_stems,
    }

    if return_model:
        return records, cameras, images, points3D, audit
    return records


def evenly_sample_view_positions(train_pool_size: int, k: int) -> List[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k > train_pool_size:
        raise ValueError(f"k={k} > train_pool size={train_pool_size}")
    if k == 1:
        return [train_pool_size // 2]
    positions = np.linspace(0, train_pool_size - 1, k)
    positions = np.round(positions).astype(int).tolist()
    if len(set(positions)) != k:
        raise RuntimeError(
            f"even sampling produced duplicate positions: positions={positions}, "
            f"train_pool_size={train_pool_size}, k={k}"
        )
    return positions


def evenly_sample_views(train_pool: Sequence[ViewRecord], k: int) -> List[ViewRecord]:
    return [train_pool[pos] for pos in evenly_sample_view_positions(len(train_pool), k)]


def pose_fps_sample_view_positions(train_pool: Sequence[ViewRecord], k: int) -> List[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(train_pool):
        raise ValueError(f"k={k} > train_pool size={len(train_pool)}")
    if k == 1:
        return [len(train_pool) // 2]

    centers = np.asarray([v.camera_center for v in train_pool], dtype=np.float64)
    selected = [0]
    dists = np.linalg.norm(centers - centers[0][None, :], axis=1)
    while len(selected) < k:
        next_idx = int(np.argmax(dists))
        if next_idx in selected:
            remaining = [i for i in range(len(train_pool)) if i not in selected]
            next_idx = remaining[0]
        selected.append(next_idx)
        dists = np.minimum(dists, np.linalg.norm(centers - centers[next_idx][None, :], axis=1))
    return sorted(selected)


def compute_view_split(
    sorted_views: Sequence[ViewRecord],
    hold: int,
    train_views: str,
    sample_mode: str,
):
    if hold <= 0:
        raise ValueError("split_hold must be > 0")

    test_positions = set(range(0, len(sorted_views), hold))
    test_views = [v for i, v in enumerate(sorted_views) if i in test_positions]
    train_pool = [v for i, v in enumerate(sorted_views) if i not in test_positions]

    if train_views == "full":
        selected_train = list(train_pool)
        train_pool_positions = list(range(len(train_pool)))
    else:
        k = int(train_views)
        if sample_mode == "paper_even":
            train_pool_positions = evenly_sample_view_positions(len(train_pool), k)
        elif sample_mode == "pose_fps":
            train_pool_positions = pose_fps_sample_view_positions(train_pool, k)
        else:
            raise ValueError(f"unknown split_train_sample_mode: {sample_mode}")
        selected_train = [train_pool[p] for p in train_pool_positions]

    for view in selected_train:
        view.split = "train"
        view.train_pool_position = train_pool.index(view)
    for view in test_views:
        view.split = "test"
        view.train_pool_position = None

    return selected_train, test_views, train_pool, train_pool_positions


def compute_sequence_coverage(selected_train: Sequence[ViewRecord], sorted_views: Sequence[ViewRecord]):
    indices = sorted([int(v.global_index) for v in selected_train])
    span = max(indices) - min(indices) if len(indices) > 1 else 0
    full_span = max(1, len(sorted_views) - 1)
    gaps = [b - a for a, b in zip(indices[:-1], indices[1:])]
    coverage_ratio = span / full_span
    return {
        "selected_global_indices": indices,
        "index_span": int(span),
        "index_coverage_ratio": float(coverage_ratio),
        "index_gaps": [int(g) for g in gaps],
        "min_index_gap": int(min(gaps)) if gaps else 0,
        "median_index_gap": float(np.median(gaps)) if gaps else 0.0,
        "max_index_gap": int(max(gaps)) if gaps else 0,
        "train_views_are_consecutive": bool(all(g == 1 for g in gaps)) if gaps else False,
        "local_cluster_guard_pass": bool(coverage_ratio >= 0.70) if len(indices) >= 3 else True,
    }


def _bbox_diag(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extent))


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def compute_pose_audit(selected_train: Sequence[ViewRecord], sorted_views: Sequence[ViewRecord]):
    full_centers = np.asarray([v.camera_center for v in sorted_views], dtype=np.float64)
    train_centers = np.asarray([v.camera_center for v in selected_train], dtype=np.float64)
    train_bbox = _bbox_diag(train_centers)
    full_bbox = _bbox_diag(full_centers)
    bbox_ratio = float(train_bbox / full_bbox) if full_bbox > 0 else 0.0

    ordered_train = sorted(selected_train, key=lambda v: v.global_index)
    center_gaps = [
        float(np.linalg.norm(
            np.asarray(b.camera_center, dtype=np.float64) - np.asarray(a.camera_center, dtype=np.float64)
        ))
        for a, b in zip(ordered_train[:-1], ordered_train[1:])
    ]

    directions = []
    for view in ordered_train:
        qvec = np.asarray(view.qvec, dtype=np.float64)
        rot = qvec2rotmat(qvec)
        direction = rot.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norm = np.linalg.norm(direction)
        directions.append(direction / norm if norm > 0 else direction)

    direction_gaps = []
    for a, b in zip(directions[:-1], directions[1:]):
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        direction_gaps.append(float(math.degrees(math.acos(dot))))

    return {
        "enabled": True,
        "camera_center_train_bbox_diag": train_bbox,
        "camera_center_full_bbox_diag": full_bbox,
        "camera_center_bbox_coverage_ratio": bbox_ratio,
        "mean_train_camera_center_gap": _safe_mean(center_gaps),
        "median_train_camera_center_gap": _safe_median(center_gaps),
        "max_train_camera_center_gap": float(max(center_gaps)) if center_gaps else 0.0,
        "mean_view_direction_gap_deg": _safe_mean(direction_gaps),
        "median_view_direction_gap_deg": _safe_median(direction_gaps),
        "pose_coverage_guard_pass": bool(bbox_ratio >= 0.20) if len(selected_train) >= 3 else True,
    }


def view_records_to_report_rows(
    train_views: Sequence[ViewRecord],
    test_views: Sequence[ViewRecord],
) -> List[dict]:
    rows = []
    for split_name, views in (("train", train_views), ("test", test_views)):
        for view in views:
            rows.append(
                {
                    "split": split_name,
                    "global_index": int(view.global_index),
                    "train_pool_position": "" if view.train_pool_position is None else int(view.train_pool_position),
                    "colmap_image_id": int(view.colmap_image_id),
                    "camera_id": int(view.camera_id),
                    "image_name": view.image_name,
                    "image_path": view.image_path,
                    "camera_center_x": float(view.camera_center[0]),
                    "camera_center_y": float(view.camera_center[1]),
                    "camera_center_z": float(view.camera_center[2]),
                    "qvec_w": float(view.qvec[0]),
                    "qvec_x": float(view.qvec[1]),
                    "qvec_y": float(view.qvec[2]),
                    "qvec_z": float(view.qvec[3]),
                    "tvec_x": float(view.tvec[0]),
                    "tvec_y": float(view.tvec[1]),
                    "tvec_z": float(view.tvec[2]),
                }
            )
    return rows


def write_selected_views_csv(path: str, train_views: Sequence[ViewRecord], test_views: Sequence[ViewRecord]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = view_records_to_report_rows(train_views, test_views)
    fieldnames = [
        "split",
        "global_index",
        "train_pool_position",
        "colmap_image_id",
        "camera_id",
        "image_name",
        "image_path",
        "camera_center_x",
        "camera_center_y",
        "camera_center_z",
        "qvec_w",
        "qvec_x",
        "qvec_y",
        "qvec_z",
        "tvec_x",
        "tvec_y",
        "tvec_z",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_names(views: Iterable[ViewRecord]) -> List[str]:
    return [v.image_name for v in views]


def selected_ids(views: Iterable[ViewRecord]) -> List[int]:
    return [int(v.colmap_image_id) for v in views]
