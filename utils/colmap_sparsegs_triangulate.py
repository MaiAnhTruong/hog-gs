import json
import os
import shutil
import sqlite3
import subprocess
from typing import Dict, Sequence

import numpy as np

from utils.colmap_split_io import (
    images_for_views,
    points3d_dict_to_arrays,
    subset_cameras_for_images,
    write_points3D_ply,
)
from utils.read_write_model import Image, read_model, write_model
from utils.view_split_utils import (
    load_full_colmap_model,
    natural_lower_key,
    normalized_rel_name,
)


def _json_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def _write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def _resolve_colmap_exe(colmap_exe: str) -> str:
    if colmap_exe and colmap_exe.lower() != "colmap":
        if os.path.isfile(colmap_exe):
            return os.path.abspath(colmap_exe)
        resolved = shutil.which(colmap_exe)
        if resolved:
            return resolved
        raise FileNotFoundError(f"COLMAP executable not found: {colmap_exe}")

    for candidate in (
        r"D:\colmap-x64-windows-cuda\bin\colmap.exe",
        "colmap.exe",
        "COLMAP.exe",
        "colmap",
        r"D:\colmap-x64-windows-cuda\COLMAP.bat",
        "COLMAP.bat",
        "colmap.bat",
    ):
        if os.path.isabs(candidate) and os.path.isfile(candidate):
            return candidate
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("COLMAP executable not found. Set --split_colmap_exe.")


def _run_colmap(colmap_exe: str, args, log_path: str, cwd: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    exe = _resolve_colmap_exe(colmap_exe)
    cmd = [exe] + list(args)
    cmd_line = subprocess.list2cmdline(cmd)
    is_batch = exe.lower().endswith((".bat", ".cmd"))
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n[COLMAP-CMD] " + cmd_line + "\n")
        proc = subprocess.run(
            cmd_line if is_batch else cmd,
            cwd=cwd,
            shell=is_batch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(proc.stdout or "")
        log.write(f"\n[COLMAP-EXIT] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"COLMAP command failed with exit code {proc.returncode}: {cmd_line}")
    return exe


def _read_database_images(database_path: str) -> Dict[str, int]:
    if not os.path.isfile(database_path):
        raise FileNotFoundError(f"COLMAP database not found: {database_path}")

    with sqlite3.connect(database_path) as conn:
        rows = conn.execute("SELECT image_id, name FROM images").fetchall()

    by_name = {}
    for image_id, name in rows:
        key = normalized_rel_name(name)
        by_name[key] = int(image_id)
        by_name[os.path.basename(key)] = int(image_id)

    return by_name


def _reindex_images_to_database(images_subset: dict, train_views: Sequence, db_image_ids: Dict[str, int]):
    reindexed = {}
    missing = []
    for view in train_views:
        source_image = images_subset[int(view.colmap_image_id)]
        candidates = [
            normalized_rel_name(view.image_name),
            os.path.basename(normalized_rel_name(view.image_name)),
            normalized_rel_name(source_image.name),
            os.path.basename(normalized_rel_name(source_image.name)),
        ]
        db_id = None
        for key in candidates:
            if key in db_image_ids:
                db_id = db_image_ids[key]
                break
        if db_id is None:
            missing.append(view.image_name)
            continue

        reindexed[int(db_id)] = Image(
            id=int(db_id),
            qvec=np.asarray(source_image.qvec, dtype=np.float64).copy(),
            tvec=np.asarray(source_image.tvec, dtype=np.float64).copy(),
            camera_id=int(source_image.camera_id),
            name=source_image.name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    if missing:
        raise RuntimeError(f"selected train images missing from COLMAP database: {missing[:20]}")
    if len(reindexed) != len(train_views):
        raise RuntimeError(
            f"database image id collision while reindexing train model: "
            f"reindexed={len(reindexed)}, requested={len(train_views)}"
        )
    return dict(sorted(reindexed.items()))


def _read_any_model(path: str):
    try:
        return read_model(path, ext=".bin"), ".bin"
    except Exception:
        return read_model(path, ext=".txt"), ".txt"


def run_sparsegs_triangulate_init(
    full_source_path: str,
    split_root: str,
    train_views: Sequence,
    test_views: Sequence,
    colmap_exe: str,
    matcher: str,
    min_points: int,
    force: bool,
) -> dict:
    if matcher not in {"exhaustive", "sequential"}:
        raise ValueError(f"unknown split_colmap_matcher: {matcher}")

    full_source_path = os.path.abspath(full_source_path)
    split_root = os.path.abspath(split_root)
    train_source_path = os.path.join(split_root, "train")
    train_images_dir = os.path.join(train_source_path, "images")
    train_sparse0 = os.path.join(train_source_path, "sparse", "0")
    reports_dir = os.path.join(split_root, "reports")
    work_dir = os.path.join(split_root, "work", "colmap")
    selected_txt_dir = os.path.join(work_dir, "sparse_selected_txt")
    database_path = os.path.join(work_dir, "database.db")
    log_path = os.path.join(work_dir, "sparsegs_triangulate.log")
    report_path = os.path.join(reports_dir, "sparsegs_triangulate_report.json")

    if os.path.exists(database_path):
        if not force:
            raise RuntimeError(f"COLMAP database already exists; use --split_force: {database_path}")
        os.remove(database_path)

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(selected_txt_dir, exist_ok=True)
    os.makedirs(train_sparse0, exist_ok=True)

    full_cameras, full_images, _ = load_full_colmap_model(full_source_path)
    train_images_subset = images_for_views(full_images, train_views)
    train_cameras_subset = subset_cameras_for_images(full_cameras, train_images_subset.values())

    feature_args = [
        "feature_extractor",
        "--database_path", database_path,
        "--image_path", train_images_dir,
        "--FeatureExtraction.use_gpu", "1",
    ]
    if len(train_cameras_subset) == 1:
        camera = next(iter(train_cameras_subset.values()))
        feature_args.extend([
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", str(camera.model),
            "--ImageReader.camera_params", ",".join(str(float(p)) for p in camera.params),
        ])
    else:
        feature_args.extend(["--ImageReader.single_camera", "0"])

    resolved_colmap = _run_colmap(
        colmap_exe,
        feature_args,
        log_path=log_path,
        cwd=work_dir,
    )
    feature_extractor_status = "PASS"

    db_image_ids = _read_database_images(database_path)
    reindexed_images = _reindex_images_to_database(train_images_subset, train_views, db_image_ids)

    write_model(train_cameras_subset, reindexed_images, {}, selected_txt_dir, ext=".txt")

    matcher_command = "exhaustive_matcher" if matcher == "exhaustive" else "sequential_matcher"
    _run_colmap(
        resolved_colmap,
        [
            matcher_command,
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", "1",
        ],
        log_path=log_path,
        cwd=work_dir,
    )
    matcher_status = "PASS"

    _run_colmap(
        resolved_colmap,
        [
            "point_triangulator",
            "--database_path", database_path,
            "--image_path", train_images_dir,
            "--input_path", selected_txt_dir,
            "--output_path", train_sparse0,
        ],
        log_path=log_path,
        cwd=work_dir,
    )
    point_triangulator_status = "PASS"

    (tri_cameras, tri_images, tri_points3D), output_format = _read_any_model(train_sparse0)
    write_model(tri_cameras, tri_images, tri_points3D, train_sparse0, ext=".bin")
    xyz, rgb = points3d_dict_to_arrays(tri_points3D)
    write_points3D_ply(xyz, rgb, os.path.join(train_sparse0, "points3D.ply"))

    triangulated_point_count = int(len(tri_points3D))
    if triangulated_point_count < int(min_points):
        raise RuntimeError(
            f"triangulated point count too low: {triangulated_point_count} < {int(min_points)}"
        )

    requested_names = sorted([normalized_rel_name(v.image_name) for v in train_views], key=natural_lower_key)
    output_names = sorted([normalized_rel_name(img.name) for img in tri_images.values()], key=natural_lower_key)
    missing_output = sorted(set(requested_names) - set(output_names), key=natural_lower_key)
    if missing_output:
        raise RuntimeError(f"point_triangulator output is missing selected train images: {missing_output[:20]}")

    report = {
        "status": "PASS",
        "policy": "sparsegs_triangulate",
        "colmap_exe": _json_path(resolved_colmap),
        "matcher": matcher,
        "database_path": _json_path(database_path),
        "work_dir": _json_path(work_dir),
        "log_path": _json_path(log_path),
        "selected_model_path": _json_path(selected_txt_dir),
        "output_sparse_path": _json_path(train_sparse0),
        "output_model_format_detected": output_format,
        "selected_train_view_count": int(len(train_views)),
        "test_view_count": int(len(test_views)),
        "feature_extractor_status": feature_extractor_status,
        "matcher_status": matcher_status,
        "point_triangulator_status": point_triangulator_status,
        "triangulated_camera_count": int(len(tri_cameras)),
        "triangulated_image_count": int(len(tri_images)),
        "triangulated_point_count": triangulated_point_count,
        "min_required_points": int(min_points),
        "full_points3D_used_for_train_initialization": False,
        "train_points_source": "point_triangulator_on_selected_train_views_only",
        "pose_source": "full_colmap_reference_poses",
        "database_image_id_by_name": {
            normalized_rel_name(v.image_name): int(reindexed_images_by_id.id)
            for reindexed_images_by_id in reindexed_images.values()
            for v in train_views
            if normalized_rel_name(v.image_name) == normalized_rel_name(reindexed_images_by_id.name)
        },
    }
    _write_json(report_path, report)
    return report
