import os
import shutil
import subprocess
from typing import Dict, Sequence

import numpy as np

from utils.colmap_split_io import clone_point_without_track
from utils.read_write_model import Point3D, qvec2rotmat, read_model
from utils.view_split_utils import natural_lower_key, normalized_rel_name


def _json_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


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


def _select_colmap_model(sparse_parent: str):
    candidates = []
    if os.path.isdir(sparse_parent):
        for name in os.listdir(sparse_parent):
            path = os.path.join(sparse_parent, name)
            if not os.path.isdir(path):
                continue
            try:
                cameras, images, points3D = read_model(path, ext=".bin")
            except Exception:
                continue
            candidates.append((len(images), len(points3D), path, cameras, images, points3D))
    if not candidates:
        raise RuntimeError(f"COLMAP mapper produced no readable sparse model under: {sparse_parent}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2], candidates[0][3], candidates[0][4], candidates[0][5]


def _camera_center(image) -> np.ndarray:
    rot = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
    tvec = np.asarray(image.tvec, dtype=np.float64)
    return -rot.T @ tvec


def estimate_sim3_umeyama(src_centers, dst_centers):
    """
    Estimate x_dst = scale * R @ x_src + t.
    """
    src = np.asarray(src_centers, dtype=np.float64)
    dst = np.asarray(dst_centers, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst center shape mismatch: src={src.shape}, dst={dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 paired camera centers for Sim(3) alignment.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / src.shape[0]
    U, singular_values, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0
    rot = U @ S @ Vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ S) / max(var_src, 1e-12))
    trans = mu_dst - scale * (rot @ mu_src)

    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"invalid Sim(3) scale: {scale}")
    if not np.all(np.isfinite(rot)) or not np.all(np.isfinite(trans)):
        raise RuntimeError("invalid Sim(3) rotation/translation")

    return scale, rot, trans


def _transform_points3d(points3D: Dict[int, Point3D], scale: float, rot: np.ndarray, trans: np.ndarray):
    out = {}
    for point_id, point in points3D.items():
        xyz_src = np.asarray(point.xyz, dtype=np.float64)
        xyz_dst = scale * (rot @ xyz_src) + trans
        out[int(point_id)] = Point3D(
            id=int(point.id),
            xyz=xyz_dst,
            rgb=np.asarray(point.rgb, dtype=np.uint8).copy(),
            error=float(np.asarray(point.error).item()),
            image_ids=np.empty((0,), dtype=np.int32),
            point2D_idxs=np.empty((0,), dtype=np.int32),
        )
    return out


def _names_by_normalized(images: dict):
    return {normalized_rel_name(img.name): img for img in images.values()}


def _reference_views_by_normalized(selected_train_views: Sequence):
    return {normalized_rel_name(v.image_name): v for v in selected_train_views}


def run_train_only_colmap(
    train_images_dir: str,
    work_dir: str,
    colmap_exe: str,
    matcher: str,
    selected_train_views: Sequence,
    full_reference_views_by_name: dict,
    require_all_registered: bool = True,
    min_aligned_points: int = 100,
) -> dict:
    if matcher not in {"exhaustive", "sequential"}:
        raise ValueError(f"unknown split_colmap_matcher: {matcher}")

    train_images_dir = os.path.abspath(train_images_dir)
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    database_path = os.path.join(work_dir, "database.db")
    sparse_parent = os.path.join(work_dir, "sparse")
    log_path = os.path.join(work_dir, "colmap_train_only.log")

    resolved_colmap = _run_colmap(
        colmap_exe,
        [
            "feature_extractor",
            "--database_path", database_path,
            "--image_path", train_images_dir,
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "1",
        ],
        log_path=log_path,
        cwd=work_dir,
    )

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

    requested_names = [normalized_rel_name(v.image_name) for v in selected_train_views]
    requested_set = set(requested_names)
    mapper_attempts = []
    selected_model_path = None
    raw_cameras = raw_images = raw_points3D = None
    best_missing = sorted(requested_set, key=natural_lower_key)

    mapper_configs = [
        (
            "default",
            sparse_parent,
            ["--Mapper.min_model_size", "2"],
        ),
        (
            "relaxed",
            os.path.join(work_dir, "sparse_relaxed"),
            [
                "--Mapper.min_model_size", "2",
                "--Mapper.min_num_matches", "5",
                "--Mapper.init_min_num_inliers", "8",
                "--Mapper.abs_pose_min_num_inliers", "6",
                "--Mapper.abs_pose_min_inlier_ratio", "0.05",
                "--Mapper.init_min_tri_angle", "4",
                "--Mapper.ba_local_min_tri_angle", "2",
                "--Mapper.filter_min_tri_angle", "0.5",
                "--Mapper.tri_min_angle", "0.5",
                "--Mapper.tri_ignore_two_view_tracks", "0",
            ],
        ),
        (
            "very_relaxed_sparse12",
            os.path.join(work_dir, "sparse_very_relaxed"),
            [
                "--Mapper.min_model_size", "2",
                "--Mapper.min_num_matches", "3",
                "--Mapper.init_min_num_inliers", "6",
                "--Mapper.abs_pose_min_num_inliers", "4",
                "--Mapper.abs_pose_min_inlier_ratio", "0.03",
                "--Mapper.init_min_tri_angle", "1.5",
                "--Mapper.ba_local_min_tri_angle", "1.0",
                "--Mapper.filter_min_tri_angle", "0.25",
                "--Mapper.tri_min_angle", "0.25",
                "--Mapper.tri_ignore_two_view_tracks", "0",
            ],
        ),
    ]

    for attempt_name, attempt_sparse_parent, mapper_args in mapper_configs:
        os.makedirs(attempt_sparse_parent, exist_ok=True)
        attempt_report = {
            "name": attempt_name,
            "output_path": _json_path(attempt_sparse_parent),
            "status": "PENDING",
        }
        try:
            _run_colmap(
                resolved_colmap,
                [
                    "mapper",
                    "--database_path", database_path,
                    "--image_path", train_images_dir,
                    "--output_path", attempt_sparse_parent,
                    *mapper_args,
                ],
                log_path=log_path,
                cwd=work_dir,
            )
            attempt_model_path, attempt_cameras, attempt_images, attempt_points3D = _select_colmap_model(attempt_sparse_parent)
            attempt_by_name = _names_by_normalized(attempt_images)
            attempt_missing = sorted(requested_set - set(attempt_by_name), key=natural_lower_key)
            attempt_report.update(
                {
                    "status": "PASS",
                    "selected_model_path": _json_path(attempt_model_path),
                    "registered_train_view_count": len(attempt_images),
                    "raw_point_count": len(attempt_points3D),
                    "missing_registered_train_views": attempt_missing,
                }
            )

            better = (
                raw_images is None
                or len(attempt_missing) < len(best_missing)
                or (
                    len(attempt_missing) == len(best_missing)
                    and len(attempt_points3D) > len(raw_points3D)
                )
            )
            if better:
                selected_model_path = attempt_model_path
                raw_cameras = attempt_cameras
                raw_images = attempt_images
                raw_points3D = attempt_points3D
                best_missing = attempt_missing

            mapper_attempts.append(attempt_report)
            if len(attempt_missing) == 0 and len(attempt_images) >= len(selected_train_views):
                break
        except Exception as exc:
            attempt_report.update({"status": "FAIL", "error": str(exc)})
            mapper_attempts.append(attempt_report)

    if raw_images is None:
        raise RuntimeError(f"COLMAP mapper produced no usable train-only model. attempts={mapper_attempts}")

    raw_by_name = _names_by_normalized(raw_images)
    registered_set = set(raw_by_name)
    missing_registered = sorted(requested_set - registered_set, key=natural_lower_key)
    extra_registered = sorted(registered_set - requested_set, key=natural_lower_key)
    all_train_views_registered = len(missing_registered) == 0 and len(raw_images) >= len(selected_train_views)

    if require_all_registered and not all_train_views_registered:
        raise RuntimeError(
            "train_only_colmap did not register all selected train views. "
            f"registered={len(raw_images)}, requested={len(selected_train_views)}, "
            f"missing={missing_registered[:20]}"
        )

    reference_by_name = {
        normalized_rel_name(name): view
        for name, view in full_reference_views_by_name.items()
    }
    selected_ref_by_name = _reference_views_by_normalized(selected_train_views)
    common_names = sorted(requested_set & registered_set & set(reference_by_name), key=natural_lower_key)
    if len(common_names) < 3:
        raise RuntimeError(
            f"Not enough common views for train-only to full-reference Sim(3): {len(common_names)}"
        )

    src_centers = np.stack([_camera_center(raw_by_name[name]) for name in common_names], axis=0)
    dst_centers = np.stack(
        [np.asarray(reference_by_name[name].camera_center, dtype=np.float64) for name in common_names],
        axis=0,
    )
    scale, rot, trans = estimate_sim3_umeyama(src_centers, dst_centers)
    aligned_centers = np.asarray([scale * (rot @ c) + trans for c in src_centers], dtype=np.float64)
    center_errors = np.linalg.norm(aligned_centers - dst_centers, axis=1)

    aligned_points3D = _transform_points3d(raw_points3D, scale, rot, trans)
    aligned_point_count = len(aligned_points3D)
    if aligned_point_count < int(min_aligned_points):
        raise RuntimeError(
            f"aligned train-only point count too low: {aligned_point_count} < {min_aligned_points}"
        )

    report = {
        "status": "PASS",
        "colmap_exe": _json_path(resolved_colmap),
        "database_path": _json_path(database_path),
        "work_dir": _json_path(work_dir),
        "log_path": _json_path(log_path),
        "selected_model_path": _json_path(selected_model_path),
        "requested_train_view_count": len(selected_train_views),
        "registered_train_view_count": len(raw_images),
        "all_train_views_registered": bool(all_train_views_registered),
        "missing_registered_train_views": missing_registered,
        "extra_registered_views": extra_registered,
        "registered_image_names": sorted([img.name for img in raw_images.values()], key=natural_lower_key),
        "raw_camera_count": len(raw_cameras),
        "raw_point_count": len(raw_points3D),
        "aligned_point_count": aligned_point_count,
        "mapper_attempts": mapper_attempts,
        "alignment_method": "umeyama_camera_centers_train_only_to_full_reference",
        "num_alignment_views": len(common_names),
        "alignment_image_names": common_names,
        "sim3_scale": float(scale),
        "sim3_rotation": rot.tolist(),
        "sim3_rotation_det": float(np.linalg.det(rot)),
        "sim3_translation": trans.tolist(),
        "center_rmse_after_alignment": float(np.sqrt(np.mean(center_errors ** 2))),
        "center_median_error_after_alignment": float(np.median(center_errors)),
        "center_max_error_after_alignment": float(np.max(center_errors)),
        "pose_protocol": "posed_sparse_view_nvs",
        "pose_source": "full_colmap_reference_for_train_and_test_camera_poses",
        "initial_point_cloud_source": "train_only_colmap_selected_train_views",
        "test_rgb_used_for_training": False,
        "test_points_used_for_initialization": False,
        "strict_rgb_and_points_from_train_views": True,
        "all_camera_poses_assumed_known": True,
    }

    for key in ("center_rmse_after_alignment", "center_median_error_after_alignment", "center_max_error_after_alignment"):
        if not np.isfinite(report[key]):
            raise RuntimeError(f"non-finite alignment metric: {key}={report[key]}")

    return {
        "status": "PASS",
        "report": report,
        "aligned_points3D": aligned_points3D,
        "raw_cameras": raw_cameras,
        "raw_images": raw_images,
        "raw_points3D": raw_points3D,
    }
