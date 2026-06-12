#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import re
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    split_manifest: dict

def canonical_image_id(image_name: str) -> str:
    """
    Normalize image name for overlap checking.
    Example:
        DSCF0656.JPG -> dscf0656
        images/DSCF0656.jpg -> dscf0656
    """
    return Path(str(image_name).replace("\\", "/")).stem.lower()


def natural_sort_key(s: str):
    """
    Natural lowercase sort:
        img2 < img10
    """
    stem = canonical_image_id(s)
    return [
        int(t) if t.isdigit() else t
        for t in re.split(r"(\d+)", stem)
    ]

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def _sparse_split_requested(sparse_train_images="", sparse_train_indices="", sparse_train_count=0):
    return bool(sparse_train_images) or bool(sparse_train_indices) or sparse_train_count > 0


def _read_colmap_extrinsics_intrinsics_from_sparse(path, sparse_rel="sparse/0", error_prefix="[BASE-SPLIT][ERROR]"):
    sparse_path = os.path.join(path, sparse_rel)
    bin_extrinsic_file = os.path.join(sparse_path, "images.bin")
    bin_intrinsic_file = os.path.join(sparse_path, "cameras.bin")
    txt_extrinsic_file = os.path.join(sparse_path, "images.txt")
    txt_intrinsic_file = os.path.join(sparse_path, "cameras.txt")

    has_bin = os.path.exists(bin_extrinsic_file) and os.path.exists(bin_intrinsic_file)
    has_txt = os.path.exists(txt_extrinsic_file) and os.path.exists(txt_intrinsic_file)
    if not has_bin and not has_txt:
        raise FileNotFoundError(
            f"{error_prefix} full eval sparse cameras/images not found: {sparse_path}"
        )

    if has_bin:
        try:
            cameras_extrinsic_file = bin_extrinsic_file
            cameras_intrinsic_file = bin_intrinsic_file
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            sparse_format = "bin"
            return cam_extrinsics, cam_intrinsics, sparse_format
        except Exception:
            if not has_txt:
                raise

    if has_txt:
        cameras_extrinsic_file = txt_extrinsic_file
        cameras_intrinsic_file = txt_intrinsic_file
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        sparse_format = "txt"

    return cam_extrinsics, cam_intrinsics, sparse_format


def _read_colmap_extrinsics_intrinsics(path):
    return _read_colmap_extrinsics_intrinsics_from_sparse(
        path,
        sparse_rel="sparse/0",
        error_prefix="[COLMAP][ERROR]",
    )


def _read_name_list_txt(path):
    if path is None or path == "":
        return []
    with open(path, "r", encoding="utf-8") as f:
        names = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(os.path.basename(line.replace("\\", "/")))
    return names


def _camera_name_keys(image_name):
    normalized = os.path.basename(str(image_name).strip().replace("\\", "/"))
    stem = Path(normalized).stem
    return {normalized, normalized.lower(), stem, stem.lower()}


def _camera_name_key_set(cam_infos):
    keys = set()
    for cam_info in cam_infos:
        keys.update(_camera_name_keys(cam_info.image_name))
    return keys


def _read_sparse_train_image_names(value, scene_path):
    if not value:
        return []

    candidate_paths = [value]
    if not os.path.isabs(value):
        candidate_paths.extend([
            os.path.join(scene_path, value),
            os.path.join(scene_path, "sparse/0", value),
        ])

    list_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
    if list_path is None:
        return [token.strip() for token in value.split(",") if token.strip()]

    if list_path.lower().endswith(".json"):
        with open(list_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            for key in ("train", "train_images", "images", "frames"):
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"Sparse train image JSON must contain a list: {list_path}")

        names = []
        for item in data:
            if isinstance(item, dict):
                name = item.get("image_name") or item.get("file_name") or item.get("file_path") or item.get("path")
            else:
                name = item
            if name:
                names.append(str(name).strip())
        return names

    with open(list_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]


def _parse_sparse_train_indices(value, num_cameras):
    if not value:
        return []

    indices = []
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue

        if ":" in token:
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid sparse train index range: {token}")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            indices.extend(range(start, stop, step))
        elif "-" in token and not token.startswith("-"):
            start, stop = [int(v) for v in token.split("-", 1)]
            step = 1 if stop >= start else -1
            indices.extend(range(start, stop + step, step))
        else:
            indices.append(int(token))

    normalized = []
    seen = set()
    for idx in indices:
        if idx < 0:
            idx = num_cameras + idx
        if idx < 0 or idx >= num_cameras:
            raise IndexError(f"Sparse train index {idx} is out of range for {num_cameras} cameras")
        if idx not in seen:
            normalized.append(idx)
            seen.add(idx)
    return normalized


def _evenly_spaced_sparse_indices(num_cameras, sparse_train_count):
    if sparse_train_count <= 0:
        return []
    if sparse_train_count > num_cameras:
        raise ValueError(
            f"sparse_train_count={sparse_train_count} is larger than the number of cameras ({num_cameras})"
        )
    return np.linspace(0, num_cameras - 1, sparse_train_count, dtype=int).tolist()


def _build_sparse_train_test_split(all_cam_infos, scene_path, sparse_train_images="", sparse_train_indices="", sparse_train_count=0):
    selected_train = []
    selected_names = set()

    def add_train_camera(cam_info):
        if cam_info.image_name not in selected_names:
            selected_train.append(cam_info)
            selected_names.add(cam_info.image_name)

    sparse_names = _read_sparse_train_image_names(sparse_train_images, scene_path)
    if sparse_names:
        unmatched = []
        for name in sparse_names:
            name_keys = _camera_name_keys(name)
            matches = [cam for cam in all_cam_infos if _camera_name_keys(cam.image_name) & name_keys]
            if not matches:
                unmatched.append(name)
            for cam in matches:
                add_train_camera(cam)
        if unmatched:
            print("[WARN] Sparse train image names not found: {}".format(", ".join(unmatched)))
    elif sparse_train_indices:
        for idx in _parse_sparse_train_indices(sparse_train_indices, len(all_cam_infos)):
            add_train_camera(all_cam_infos[idx])
    elif sparse_train_count > 0:
        for idx in _evenly_spaced_sparse_indices(len(all_cam_infos), sparse_train_count):
            add_train_camera(all_cam_infos[idx])

    if not selected_train:
        raise ValueError("Sparse-view split was requested, but no train cameras were selected.")

    train_name_set = {cam.image_name for cam in selected_train}
    train_cam_infos = [cam._replace(is_test=False) for cam in selected_train]
    test_cam_infos = [cam._replace(is_test=True) for cam in all_cam_infos if cam.image_name not in train_name_set]

    print(
        "[SPARSE-VIEW SPLIT] full={} train={} test={} (test = full - train)".format(
            len(all_cam_infos), len(train_cam_infos), len(test_cam_infos)
        )
    )
    if len(test_cam_infos) == 0:
        print("[WARN] Sparse-view split produced an empty test set.")

    return train_cam_infos, test_cam_infos


def _build_full_source_test_split(train_cam_infos, full_cam_infos):
    train_name_keys = _camera_name_key_set(train_cam_infos)
    test_cam_infos = [
        cam._replace(is_test=True)
        for cam in full_cam_infos
        if not (_camera_name_keys(cam.image_name) & train_name_keys)
    ]

    print(
        "[FULL TEST SPLIT] train={} full_test_source={} test={} (test = full_test_source - train names)".format(
            len(train_cam_infos), len(full_cam_infos), len(test_cam_infos)
        )
    )
    if len(test_cam_infos) == 0:
        print("[WARN] Full-test split produced an empty test set.")

    return test_cam_infos


def _select_eval_test_names(eval_source_path, eval_extrinsics, split_mode, llffhold, test_view_list_path):
    all_names = sorted([eval_extrinsics[k].name for k in eval_extrinsics])

    if split_mode == "llffhold":
        if llffhold <= 0:
            raise ValueError("--dpcr_eval_llffhold must be > 0 when split_mode='llffhold'")
        test_names = [name for idx, name in enumerate(all_names) if idx % llffhold == 0]
    elif split_mode == "test_txt":
        test_txt = os.path.join(eval_source_path, "sparse/0", "test.txt")
        if not os.path.exists(test_txt):
            raise FileNotFoundError(f"Expected test split file not found: {test_txt}")
        test_names = _read_name_list_txt(test_txt)
    elif split_mode == "manifest":
        if not test_view_list_path:
            raise ValueError("--dpcr_eval_test_view_list is required when split_mode='manifest'")
        test_names = _read_name_list_txt(test_view_list_path)
    else:
        raise ValueError(f"Unknown --dpcr_eval_split_mode: {split_mode}")

    all_name_set = {os.path.basename(n) for n in all_names}
    missing = sorted(set(os.path.basename(n) for n in test_names) - all_name_set)
    if missing:
        raise ValueError(
            f"Some requested eval test images are not present in eval source COLMAP images: {missing[:20]}"
        )

    return [os.path.basename(n) for n in test_names], [os.path.basename(n) for n in all_names]


def _filter_cam_infos_by_names(cam_infos, include_names=None, exclude_names=None):
    include_set = set(os.path.basename(n) for n in include_names) if include_names else None
    exclude_set = set(os.path.basename(n) for n in exclude_names) if exclude_names else set()

    out = []
    for c in cam_infos:
        name = os.path.basename(c.image_name)
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        out.append(c)
    return out


def _check_same_frame_by_common_cameras(train_cam_infos, eval_cam_infos, min_common=4, tol=1e-3):
    train_by_name = {os.path.basename(c.image_name): c for c in train_cam_infos}
    eval_by_name = {os.path.basename(c.image_name): c for c in eval_cam_infos}

    common = sorted(set(train_by_name.keys()) & set(eval_by_name.keys()))

    if len(common) < min_common:
        raise ValueError(
            f"Not enough common camera names to verify coordinate frame. "
            f"common={len(common)}, required={min_common}. "
            "Use --dpcr_eval_frame_mode skip only if you are absolutely sure both folders share the same coordinate frame."
        )

    train_centers = np.stack([_camera_center_from_caminfo(train_by_name[n]) for n in common], axis=0)
    eval_centers = np.stack([_camera_center_from_caminfo(eval_by_name[n]) for n in common], axis=0)

    scale_ref = max(np.linalg.norm(train_centers.max(axis=0) - train_centers.min(axis=0)), 1e-8)
    rmse = np.sqrt(np.mean(np.sum((train_centers - eval_centers) ** 2, axis=1)))
    rel_rmse = rmse / scale_ref

    if rel_rmse > tol:
        raise ValueError(
            "Sparse train source and eval source do not appear to share the same coordinate frame.\n"
            f"common_images={len(common)}, center_rmse={rmse}, relative_rmse={rel_rmse}, tol={tol}\n"
            "This usually happens when the 12-view folder was reconstructed separately by VGGT/SfM. "
            "Use --dpcr_eval_frame_mode align_umeyama, or regenerate the sparse folder by subsetting cameras from the full source coordinate frame."
        )

    return {
        "common_count": len(common),
        "center_rmse": float(rmse),
        "relative_rmse": float(rel_rmse),
        "mode": "strict",
    }


def _umeyama_similarity(src, dst, with_scale=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    assert src.shape == dst.shape
    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for Umeyama alignment.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)

    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1

    R = U @ D @ Vt

    if with_scale:
        var_src = np.mean(np.sum(src_c ** 2, axis=1))
        scale = np.trace(np.diag(S) @ D) / max(var_src, 1e-12)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)

    return float(scale), R.astype(np.float64), t.astype(np.float64)


def _transform_camera_info_sim3(cam, scale, R_align, t_align):
    Rwc_old = cam.R
    tcw_old = cam.T

    C_old = -Rwc_old @ tcw_old
    C_new = scale * (R_align @ C_old) + t_align
    Rwc_new = R_align @ Rwc_old
    Rcw_new = Rwc_new.T
    tcw_new = -Rcw_new @ C_new

    return CameraInfo(
        uid=cam.uid,
        R=Rwc_new.astype(np.float32),
        T=tcw_new.astype(np.float32),
        FovY=cam.FovY,
        FovX=cam.FovX,
        depth_params=cam.depth_params,
        image_path=cam.image_path,
        image_name=cam.image_name,
        depth_path=cam.depth_path,
        width=cam.width,
        height=cam.height,
        is_test=cam.is_test,
    )


def _align_eval_cameras_to_train_frame(train_cam_infos, eval_all_cam_infos, test_cam_infos, min_common=4):
    train_by_name = {os.path.basename(c.image_name): c for c in train_cam_infos}
    eval_by_name = {os.path.basename(c.image_name): c for c in eval_all_cam_infos}

    common = sorted(set(train_by_name.keys()) & set(eval_by_name.keys()))

    if len(common) < min_common:
        raise ValueError(
            f"Not enough common images for Umeyama alignment: common={len(common)}, required={min_common}. "
            "Cannot align full eval cameras to sparse train frame."
        )

    src_eval = np.stack([_camera_center_from_caminfo(eval_by_name[n]) for n in common], axis=0)
    dst_train = np.stack([_camera_center_from_caminfo(train_by_name[n]) for n in common], axis=0)

    scale, R_align, t_align = _umeyama_similarity(src_eval, dst_train, with_scale=True)

    aligned_test = [
        _transform_camera_info_sim3(c, scale, R_align, t_align)
        for c in test_cam_infos
    ]

    aligned_eval_common = [
        _transform_camera_info_sim3(eval_by_name[n], scale, R_align, t_align)
        for n in common
    ]

    aligned_by_name = {os.path.basename(c.image_name): c for c in aligned_eval_common}

    errors = []
    for n in common:
        c_train = _camera_center_from_caminfo(train_by_name[n])
        c_eval_aligned = _camera_center_from_caminfo(aligned_by_name[n])
        errors.append(np.linalg.norm(c_train - c_eval_aligned))

    rmse = float(np.sqrt(np.mean(np.square(errors))))

    return aligned_test, {
        "mode": "align_umeyama",
        "common_count": len(common),
        "scale": float(scale),
        "R": R_align.tolist(),
        "t": t_align.tolist(),
        "alignment_center_rmse": rmse,
        "common_names": common,
    }


def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    test_cam_name_set = set(os.path.basename(name) for name in test_cam_names_list)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=os.path.basename(image_name) in test_cam_name_set)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_image_files(images_folder):
    if not os.path.isdir(images_folder):
        raise FileNotFoundError(f"[BASE-SPLIT][ERROR] full eval images folder not found: {images_folder}")

    image_files = []
    for root, _, files in os.walk(images_folder):
        for filename in files:
            if Path(filename).suffix.lower() in _IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, filename))
    return image_files


def read_full_eval_cam_infos(full_eval_path, full_eval_images, full_eval_sparse):
    """
    Read COLMAP cameras/images from full_eval_path.
    Do not read point cloud for training.
    Return sorted full camera infos.
    """
    eval_extrinsics, eval_intrinsics, sparse_format = _read_colmap_extrinsics_intrinsics_from_sparse(
        full_eval_path,
        sparse_rel=full_eval_sparse,
        error_prefix="[BASE-SPLIT][ERROR]",
    )
    eval_images_folder = os.path.join(full_eval_path, full_eval_images)
    full_cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=eval_extrinsics,
        cam_intrinsics=eval_intrinsics,
        depths_params=None,
        images_folder=eval_images_folder,
        depths_folder="",
        test_cam_names_list=[],
    )
    full_cam_infos = sorted(full_cam_infos_unsorted, key=lambda c: natural_sort_key(c.image_name))
    return full_cam_infos, sparse_format


def _camera_ids(cam_infos):
    return {canonical_image_id(c.image_name) for c in cam_infos}


def _camera_center_from_caminfo(cam):
    return -cam.R @ cam.T


def _camera_scene_radius(cam_infos):
    if not cam_infos:
        return 0.0
    centers = np.stack([_camera_center_from_caminfo(c) for c in cam_infos], axis=0)
    center = np.mean(centers, axis=0, keepdims=True)
    distances = np.linalg.norm(centers - center, axis=1)
    return float(np.max(distances) * 1.1) if distances.size else 0.0


def _rotation_angle_deg(rot_a, rot_b):
    rel = np.asarray(rot_a, dtype=np.float64).T @ np.asarray(rot_b, dtype=np.float64)
    cos_theta = (np.trace(rel) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _index_by_canonical_id(cam_infos):
    by_id = {}
    duplicates = []
    for cam in cam_infos:
        image_id = canonical_image_id(cam.image_name)
        if image_id in by_id:
            duplicates.append(image_id)
            continue
        by_id[image_id] = cam
    return by_id, sorted(set(duplicates), key=natural_sort_key)


def _audit_pose_frame(train_cam_infos, full_cam_infos):
    train_by_id, train_duplicate_ids = _index_by_canonical_id(train_cam_infos)
    full_by_id, full_duplicate_ids = _index_by_canonical_id(full_cam_infos)

    train_ids = set(train_by_id.keys())
    full_ids = set(full_by_id.keys())
    common_ids = sorted(train_ids & full_ids, key=natural_sort_key)
    missing_train_ids = sorted(train_ids - full_ids, key=natural_sort_key)

    report = {
        "status": "FAIL",
        "common_image_count": len(common_ids),
        "train_image_missing_in_full_count": len(missing_train_ids),
        "train_image_missing_in_full": missing_train_ids,
        "train_duplicate_id_count": len(train_duplicate_ids),
        "full_duplicate_id_count": len(full_duplicate_ids),
    }

    if train_duplicate_ids or full_duplicate_ids:
        report["reason"] = "duplicate_canonical_image_ids"
        report["train_duplicate_ids"] = train_duplicate_ids[:50]
        report["full_duplicate_ids"] = full_duplicate_ids[:50]
        return report

    if missing_train_ids:
        report["reason"] = "train_images_not_found_in_full_eval_sparse"
        return report

    if not common_ids:
        report["reason"] = "no_common_train_full_camera_names"
        return report

    center_distances = []
    rotation_angles = []
    for image_id in common_ids:
        train_cam = train_by_id[image_id]
        full_cam = full_by_id[image_id]
        train_center = _camera_center_from_caminfo(train_cam)
        full_center = _camera_center_from_caminfo(full_cam)
        center_distances.append(float(np.linalg.norm(train_center - full_center)))
        rotation_angles.append(_rotation_angle_deg(train_cam.R, full_cam.R))

    train_scene_radius = _camera_scene_radius(train_cam_infos)
    center_tol = max(1e-5, 1e-4 * max(train_scene_radius, 1e-8))
    rotation_median_tol_deg = 0.01

    median_center = float(np.median(center_distances))
    max_center = float(np.max(center_distances))
    median_rotation = float(np.median(rotation_angles))
    max_rotation = float(np.max(rotation_angles))

    report.update({
        "median_center_distance_raw": median_center,
        "max_center_distance_raw": max_center,
        "rotation_angle_median_deg": median_rotation,
        "rotation_angle_max_deg": max_rotation,
        "center_distance_median_tolerance": center_tol,
        "rotation_angle_median_tolerance_deg": rotation_median_tol_deg,
    })

    if median_center < center_tol and median_rotation < rotation_median_tol_deg:
        report["status"] = "PASS"
        report["note"] = "train/full camera frames appear compatible"
    else:
        report["reason"] = "train_sparse_and_full_sparse_camera_frames_do_not_match"

    return report


def _select_full_eval_holdout_cameras(
    train_cam_infos,
    full_cam_infos,
    eval_hold,
    eval_overlap_shift,
    eval_boundary_forward_fallback,
    eval_strict_backward_shift,
):
    if eval_hold <= 0:
        raise ValueError("[BASE-SPLIT][ERROR] eval_hold must be > 0")
    if eval_overlap_shift not in {"none", "backward"}:
        raise ValueError("[BASE-SPLIT][ERROR] eval_overlap_shift must be one of: none, backward")

    train_ids = _camera_ids(train_cam_infos)
    used_test_ids = set()
    selected = []
    records = []
    raw_indices = list(range(0, len(full_cam_infos), eval_hold))

    raw_overlap_count = 0
    backward_shift_count = 0
    boundary_forward_fallback_count = 0
    skipped_count = 0

    def can_use(index):
        image_id = canonical_image_id(full_cam_infos[index].image_name)
        return image_id not in train_ids and image_id not in used_test_ids

    def accept(raw_index, selected_index, reason):
        nonlocal backward_shift_count, boundary_forward_fallback_count
        cam = full_cam_infos[selected_index]
        image_id = canonical_image_id(cam.image_name)
        selected.append(cam._replace(is_test=True))
        used_test_ids.add(image_id)
        if selected_index < raw_index:
            backward_shift_count += 1
        if reason == "boundary_forward_fallback_after_backward_impossible":
            boundary_forward_fallback_count += 1
        records.append({
            "raw_index": raw_index,
            "raw_image_name": full_cam_infos[raw_index].image_name,
            "raw_image_id": canonical_image_id(full_cam_infos[raw_index].image_name),
            "selected_index": selected_index,
            "selected_image_name": cam.image_name,
            "selected_image_id": image_id,
            "reason": reason,
        })

    for raw_index in raw_indices:
        raw_id = canonical_image_id(full_cam_infos[raw_index].image_name)
        raw_overlaps_train = raw_id in train_ids
        raw_is_duplicate_test = raw_id in used_test_ids
        if raw_overlaps_train:
            raw_overlap_count += 1

        if not raw_overlaps_train and not raw_is_duplicate_test:
            accept(raw_index, raw_index, "raw_candidate")
            continue

        selected_index = None
        selected_reason = None
        if eval_overlap_shift == "backward":
            bin_start = raw_index - eval_hold + 1 if raw_index > 0 else 0
            for candidate_index in range(raw_index - 1, bin_start - 1, -1):
                if can_use(candidate_index):
                    selected_index = candidate_index
                    selected_reason = "backward_shift_after_overlap_or_duplicate"
                    break

            if (
                selected_index is None
                and raw_index == 0
                and eval_boundary_forward_fallback
                and not eval_strict_backward_shift
            ):
                boundary_end = min(eval_hold, len(full_cam_infos))
                for candidate_index in range(1, boundary_end):
                    if can_use(candidate_index):
                        selected_index = candidate_index
                        selected_reason = "boundary_forward_fallback_after_backward_impossible"
                        break

        if selected_index is not None:
            accept(raw_index, selected_index, selected_reason)
        else:
            skipped_count += 1
            records.append({
                "raw_index": raw_index,
                "raw_image_name": full_cam_infos[raw_index].image_name,
                "raw_image_id": raw_id,
                "selected_index": None,
                "selected_image_name": None,
                "selected_image_id": None,
                "reason": "skipped_no_nonoverlap_candidate_in_hold_bin",
            })

    counters = {
        "raw_candidate_count": len(raw_indices),
        "raw_overlap_count": raw_overlap_count,
        "backward_shift_count": backward_shift_count,
        "boundary_forward_fallback_count": boundary_forward_fallback_count,
        "forward_fallback_count": 0,
        "skipped_count": skipped_count,
    }

    return selected, counters, records


def _write_base_split_report(report, model_path):
    if not model_path:
        return None

    os.makedirs(model_path, exist_ok=True)
    json_path = os.path.join(model_path, "split_report.json")
    txt_path = os.path.join(model_path, "split_report.txt")
    report["split_report_path"] = json_path
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        f"status: {report.get('status')}",
        f"mode: {report.get('mode')}",
        f"source_train_path: {report.get('source_train_path')}",
        f"full_eval_path: {report.get('full_eval_path')}",
        f"sort_mode: {report.get('sort_mode')}",
        f"train_count: {report.get('train_count')}",
        f"full_camera_count: {report.get('full_camera_count')}",
        f"eval_hold: {report.get('eval_hold')}",
        f"raw_candidate_count: {report.get('raw_candidate_count')}",
        f"test_count: {report.get('test_count')}",
        f"final_overlap_count: {report.get('final_overlap_count')}",
        f"duplicate_test_count: {report.get('duplicate_test_count')}",
        f"point_cloud_source: {report.get('point_cloud_source')}",
        f"nerf_normalization_source: {report.get('nerf_normalization_source')}",
        f"full_scene_point_cloud_used_for_training: {report.get('full_scene_point_cloud_used_for_training')}",
        f"pose_frame_audit.status: {report.get('pose_frame_audit', {}).get('status')}",
        f"split_report: {json_path}",
    ]
    if report.get("status") != "PASS":
        lines.append(f"failure_reasons: {', '.join(report.get('failure_reasons', []))}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path


def _emit_base_split_log(report):
    print(f"[BASE-SPLIT] status={report.get('status')}")
    print(f"[BASE-SPLIT] train_count={report.get('train_count')}")
    print(f"[BASE-SPLIT] full_camera_count={report.get('full_camera_count')}")
    print(
        "[BASE-SPLIT] eval_hold={} raw_candidate_count={} test_count={}".format(
            report.get("eval_hold"),
            report.get("raw_candidate_count"),
            report.get("test_count"),
        )
    )
    print(
        "[BASE-SPLIT] final_overlap_count={} duplicate_test_count={}".format(
            report.get("final_overlap_count"),
            report.get("duplicate_test_count"),
        )
    )
    print(f"[BASE-SPLIT] point_cloud_source={report.get('point_cloud_source')}")
    print(f"[BASE-SPLIT] nerf_normalization_source={report.get('nerf_normalization_source')}")
    if report.get("split_report_path"):
        print(f"[BASE-SPLIT] split_report={report.get('split_report_path')}")


def _finalize_base_split_report(report, split_report_enable, model_path):
    assertions = {
        "train_count_positive": report.get("train_count", 0) > 0,
        "test_count_positive": report.get("test_count", 0) > 0,
        "final_overlap_zero": report.get("final_overlap_count", 0) == 0,
        "duplicate_test_zero": report.get("duplicate_test_count", 0) == 0,
        "point_cloud_not_from_full_scene": report.get("full_scene_point_cloud_used_for_training") is False,
        "normalization_train_only": report.get("nerf_normalization_source") == "train_cameras_only",
        "pose_frame_compatible": report.get("pose_frame_audit", {}).get("status") == "PASS",
    }
    report["assertions"] = assertions
    report["status"] = "PASS" if all(assertions.values()) else "FAIL"
    if report["status"] != "PASS":
        report["failure_reasons"] = [key for key, ok in assertions.items() if not ok]

    if split_report_enable:
        _write_base_split_report(report, model_path)

    _emit_base_split_log(report)

    if report["status"] != "PASS":
        reasons = ", ".join(report.get("failure_reasons", []))
        pose_reason = report.get("pose_frame_audit", {}).get("reason")
        if pose_reason:
            reasons = f"{reasons}; {pose_reason}" if reasons else pose_reason
        raise ValueError(f"[BASE-SPLIT][ERROR] split report failed: {reasons}")

    return report


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(
    path,
    images,
    depths,
    eval,
    train_test_exp,
    llffhold=8,
    sparse_train_images="",
    sparse_train_indices="",
    sparse_train_count=0,
    full_test_source_path="",
    full_test_images="",
    external_test_source_path="",
    auto_split_report_path="",
    dpcr_eval_source_path="",
    dpcr_eval_images="",
    dpcr_eval_split_mode="llffhold",
    dpcr_eval_llffhold=8,
    dpcr_train_view_list="",
    dpcr_eval_test_view_list="",
    dpcr_eval_require_disjoint=True,
    dpcr_eval_frame_mode="strict",
    dpcr_eval_alignment_min_common=4,
    dpcr_eval_frame_check_tol=1e-3,
    full_eval_path="",
    full_eval_images="images",
    full_eval_sparse="sparse/0",
    eval_hold=None,
    eval_overlap_shift="backward",
    eval_boundary_forward_fallback=True,
    eval_strict_backward_shift=False,
    split_report_enable=True,
    model_path=None,
    split_only=False,
):
    train_source_path = path
    cam_extrinsics, cam_intrinsics, train_sparse_format = _read_colmap_extrinsics_intrinsics(train_source_path)

    depth_params_file = os.path.join(train_source_path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    sparse_split_active = _sparse_split_requested(
        sparse_train_images=sparse_train_images,
        sparse_train_indices=sparse_train_indices,
        sparse_train_count=sparse_train_count,
    )
    use_auto_external_test = bool(eval and external_test_source_path)
    use_base_full_eval = bool(eval and full_eval_path) and not use_auto_external_test
    use_external_eval = bool(eval and dpcr_eval_source_path) and not use_base_full_eval and not use_auto_external_test
    if use_auto_external_test and (full_eval_path or dpcr_eval_source_path or full_test_source_path):
        raise ValueError(
            "[AUTO-SPLIT][ERROR] do not combine --external_test_source_path "
            "with --full_eval_path, --dpcr_eval_source_path, or --full_test_source_path"
        )
    if use_base_full_eval and dpcr_eval_source_path:
        raise ValueError("[BASE-SPLIT][ERROR] do not combine --full_eval_path with --dpcr_eval_source_path")
    full_test_split_active = bool(full_test_source_path) and not use_external_eval and not use_base_full_eval and not use_auto_external_test
    split_manifest = {}

    if use_auto_external_test:
        train_reading_dir = "images" if images is None else images
        train_cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            depths_params=depths_params,
            images_folder=os.path.join(train_source_path, train_reading_dir),
            depths_folder=os.path.join(train_source_path, depths) if depths != "" else "",
            test_cam_names_list=[],
        )
        train_cam_infos = sorted(
            [c._replace(is_test=False) for c in train_cam_infos_unsorted],
            key=lambda c: natural_sort_key(c.image_name),
        )

        test_extrinsics, test_intrinsics, test_sparse_format = _read_colmap_extrinsics_intrinsics(external_test_source_path)
        test_names = [test_extrinsics[k].name for k in test_extrinsics]
        test_cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=test_extrinsics,
            cam_intrinsics=test_intrinsics,
            depths_params=None,
            images_folder=os.path.join(external_test_source_path, train_reading_dir),
            depths_folder="",
            test_cam_names_list=test_names,
        )
        test_cam_infos = sorted(
            [c._replace(is_test=True) for c in test_cam_infos_unsorted],
            key=lambda c: natural_sort_key(c.image_name),
        )

        split_manifest = {}
        if auto_split_report_path and os.path.isfile(auto_split_report_path):
            with open(auto_split_report_path, "r", encoding="utf-8") as f:
                split_manifest = json.load(f)
        split_manifest.update({
            "protocol": split_manifest.get("protocol", "mipnerf360_3dgs_sparse_posed_view_split"),
            "mode": split_manifest.get("mode", "auto_full_dataset_holdout_split"),
            "split_report_path": auto_split_report_path,
            "train_source_path": os.path.abspath(train_source_path),
            "test_source_path": os.path.abspath(external_test_source_path),
            "train_sparse_format": train_sparse_format,
            "test_sparse_format": test_sparse_format,
            "train_count": len(train_cam_infos),
            "test_count": len(test_cam_infos),
            "point_cloud_source": split_manifest.get("point_cloud_source", "train_source_only"),
            "nerf_normalization_source": split_manifest.get("nerf_normalization_source", "train_cameras_only"),
            "internal_llffhold_disabled": split_manifest.get("internal_llffhold_disabled", True),
            "external_test_source_enabled": split_manifest.get("external_test_source_enabled", True),
            "init_policy": split_manifest.get("init_policy"),
            "strict_sparse_geometry": split_manifest.get("strict_sparse_geometry"),
            "train_selection_strategy": split_manifest.get("train_selection_strategy"),
        })

        print("[SPLIT-READER] external_test_source_path detected")
        print("[SPLIT-READER] internal LLFF/every-8 split disabled")
        print(f"[SPLIT-READER] train_camera_count={len(train_cam_infos)}")
        print(f"[SPLIT-READER] test_camera_count={len(test_cam_infos)}")
        print("[SPLIT-READER] nerf_normalization_source=train_cameras_only")
        print("[SPLIT-READER] point_cloud_source=train/sparse/0/points3D.ply")

    elif use_base_full_eval:
        effective_hold = int(eval_hold if eval_hold is not None else llffhold)
        train_reading_dir = "images" if images is None else images
        train_cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            depths_params=depths_params,
            images_folder=os.path.join(train_source_path, train_reading_dir),
            depths_folder=os.path.join(train_source_path, depths) if depths != "" else "",
            test_cam_names_list=[],
        )
        train_cam_infos = sorted(
            [c._replace(is_test=False) for c in train_cam_infos_unsorted],
            key=lambda c: natural_sort_key(c.image_name),
        )

        full_cam_infos, full_sparse_format = read_full_eval_cam_infos(
            full_eval_path=full_eval_path,
            full_eval_images=full_eval_images,
            full_eval_sparse=full_eval_sparse,
        )
        test_cam_infos, selection_counters, selection_records = _select_full_eval_holdout_cameras(
            train_cam_infos=train_cam_infos,
            full_cam_infos=full_cam_infos,
            eval_hold=effective_hold,
            eval_overlap_shift=eval_overlap_shift,
            eval_boundary_forward_fallback=eval_boundary_forward_fallback,
            eval_strict_backward_shift=eval_strict_backward_shift,
        )

        train_ids = _camera_ids(train_cam_infos)
        final_test_ids = [canonical_image_id(c.image_name) for c in test_cam_infos]
        final_unique_test_ids = set(final_test_ids)
        final_overlap = sorted(final_unique_test_ids & train_ids, key=natural_sort_key)
        duplicate_test_count = len(final_test_ids) - len(final_unique_test_ids)

        full_images_folder = os.path.join(full_eval_path, full_eval_images)
        full_image_files = _list_image_files(full_images_folder)
        full_image_ids = {canonical_image_id(p) for p in full_image_files}
        full_camera_ids = _camera_ids(full_cam_infos)
        pose_frame_audit = _audit_pose_frame(train_cam_infos, full_cam_infos)

        split_report = {
            "status": "PENDING",
            "mode": "full_eval_external_test",
            "source_train_path": os.path.abspath(train_source_path),
            "full_eval_path": os.path.abspath(full_eval_path),
            "sort_mode": "natural_lowercase_image_name",
            "train_sparse_format": train_sparse_format,
            "full_eval_sparse_format": full_sparse_format,
            "train_count": len(train_cam_infos),
            "full_camera_count": len(full_cam_infos),
            "full_image_file_count": len(full_image_files),
            "full_camera_without_image_count": len(full_camera_ids - full_image_ids),
            "full_image_without_camera_count": len(full_image_ids - full_camera_ids),
            "eval_hold": effective_hold,
            "effective_hold": effective_hold,
            "raw_candidate_count": selection_counters["raw_candidate_count"],
            "raw_overlap_count": selection_counters["raw_overlap_count"],
            "backward_shift_count": selection_counters["backward_shift_count"],
            "boundary_forward_fallback_count": selection_counters["boundary_forward_fallback_count"],
            "forward_fallback_count": selection_counters["forward_fallback_count"],
            "skipped_count": selection_counters["skipped_count"],
            "test_count": len(test_cam_infos),
            "final_overlap_count": len(final_overlap),
            "duplicate_test_count": duplicate_test_count,
            "point_cloud_source": "source_train_sparse",
            "nerf_normalization_source": "train_cameras_only",
            "full_scene_point_cloud_used_for_training": False,
            "pose_frame_audit": pose_frame_audit,
            "train_image_names": [c.image_name for c in train_cam_infos],
            "test_image_names": [c.image_name for c in test_cam_infos],
            "final_overlap_image_ids": final_overlap,
            "selection_records": selection_records,
        }
        split_manifest = _finalize_base_split_report(
            split_report,
            split_report_enable=split_report_enable,
            model_path=model_path,
        )

    elif use_external_eval:
        eval_extrinsics, eval_intrinsics, eval_sparse_format = _read_colmap_extrinsics_intrinsics(dpcr_eval_source_path)
        test_names, eval_all_names = _select_eval_test_names(
            eval_source_path=dpcr_eval_source_path,
            eval_extrinsics=eval_extrinsics,
            split_mode=dpcr_eval_split_mode,
            llffhold=dpcr_eval_llffhold,
            test_view_list_path=dpcr_eval_test_view_list,
        )

        all_sparse_train_names = sorted([os.path.basename(cam_extrinsics[k].name) for k in cam_extrinsics])
        if dpcr_train_view_list:
            train_names = _read_name_list_txt(dpcr_train_view_list)
        else:
            train_names = all_sparse_train_names
        train_names = list(dict.fromkeys(os.path.basename(n) for n in train_names))

        missing_train = sorted(set(train_names) - set(all_sparse_train_names))
        if missing_train:
            raise ValueError(f"Train view list contains images not found in sparse train source: {missing_train[:20]}")

        overlap = sorted(set(train_names) & set(test_names))
        if overlap and dpcr_eval_require_disjoint:
            raise ValueError(
                "Train/test leakage detected. These images are in sparse train source but also in fixed eval test split: "
                f"{overlap[:50]}\n"
                "Regenerate the sparse-view folder using only non-test images from the full dataset, "
                "or add --no_dpcr_eval_require_disjoint only for debugging."
            )

        train_reading_dir = "images" if images is None else images
        train_cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            depths_params=depths_params,
            images_folder=os.path.join(train_source_path, train_reading_dir),
            depths_folder=os.path.join(train_source_path, depths) if depths != "" else "",
            test_cam_names_list=[],
        )
        train_cam_infos = sorted(
            _filter_cam_infos_by_names(train_cam_infos_unsorted, include_names=train_names),
            key=lambda x: x.image_name
        )

        eval_reading_dir = dpcr_eval_images if dpcr_eval_images else train_reading_dir
        eval_cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=eval_extrinsics,
            cam_intrinsics=eval_intrinsics,
            depths_params=None,
            images_folder=os.path.join(dpcr_eval_source_path, eval_reading_dir),
            depths_folder="",
            test_cam_names_list=test_names,
        )
        eval_all_cam_infos = sorted(eval_cam_infos_unsorted.copy(), key=lambda x: x.image_name)
        test_cam_infos = sorted(
            _filter_cam_infos_by_names(eval_all_cam_infos, include_names=test_names),
            key=lambda x: x.image_name
        )

        if dpcr_eval_frame_mode == "strict":
            frame_report = _check_same_frame_by_common_cameras(
                train_cam_infos=train_cam_infos,
                eval_cam_infos=eval_all_cam_infos,
                min_common=dpcr_eval_alignment_min_common,
                tol=dpcr_eval_frame_check_tol,
            )
        elif dpcr_eval_frame_mode == "align_umeyama":
            test_cam_infos, frame_report = _align_eval_cameras_to_train_frame(
                train_cam_infos=train_cam_infos,
                eval_all_cam_infos=eval_all_cam_infos,
                test_cam_infos=test_cam_infos,
                min_common=dpcr_eval_alignment_min_common,
            )
        elif dpcr_eval_frame_mode == "skip":
            frame_report = {"mode": "skip", "warning": "coordinate frame check skipped"}
        else:
            raise ValueError(f"Unknown --dpcr_eval_frame_mode: {dpcr_eval_frame_mode}")

        eval_test_set = set(test_names)
        train_set = set(train_names)
        eval_all_set = set(eval_all_names)
        unused_names = sorted(eval_all_set - eval_test_set - train_set)

        split_manifest = {
            "protocol": "dpcr_sparse_train_external_eval",
            "train_source_path": os.path.abspath(train_source_path),
            "eval_source_path": os.path.abspath(dpcr_eval_source_path),
            "train_sparse_format": train_sparse_format,
            "eval_sparse_format": eval_sparse_format,
            "eval_split_mode": dpcr_eval_split_mode,
            "eval_llffhold": int(dpcr_eval_llffhold),
            "sort_key": "sorted image_name",
            "train_count": len(train_cam_infos),
            "test_count": len(test_cam_infos),
            "eval_full_count": len(eval_all_names),
            "unused_count": len(unused_names),
            "train_image_names": sorted([os.path.basename(c.image_name) for c in train_cam_infos]),
            "test_image_names": sorted([os.path.basename(c.image_name) for c in test_cam_infos]),
            "unused_image_names": unused_names,
            "overlap_train_test": sorted(train_set & eval_test_set),
            "frame_report": frame_report,
        }

        print("------------DPCR EXTERNAL EVAL SPLIT-------------")
        print(f"[SPLIT] train_source: {train_source_path}")
        print(f"[SPLIT] eval_source : {dpcr_eval_source_path}")
        print(f"[SPLIT] train_count : {len(train_cam_infos)}")
        print(f"[SPLIT] test_count  : {len(test_cam_infos)}")
        print(f"[SPLIT] unused_count: {len(unused_names)}")
        print(f"[SPLIT] overlap    : {len(split_manifest['overlap_train_test'])}")
        print(f"[SPLIT] frame_mode : {frame_report.get('mode')}")
        print("-------------------------------------------------")

    elif eval and not sparse_split_active and not full_test_split_active:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    if not use_auto_external_test and not use_external_eval and not use_base_full_eval:
        reading_dir = "images" if images == None else images
        cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
            images_folder=os.path.join(path, reading_dir),
            depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
        all_cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

        if sparse_split_active:
            train_cam_infos, test_cam_infos = _build_sparse_train_test_split(
                all_cam_infos=all_cam_infos,
                scene_path=path,
                sparse_train_images=sparse_train_images,
                sparse_train_indices=sparse_train_indices,
                sparse_train_count=sparse_train_count,
            )
            if full_test_split_active:
                test_cam_infos = []
        else:
            train_cam_infos = [c for c in all_cam_infos if train_test_exp or not c.is_test]
            test_cam_infos = [c for c in all_cam_infos if c.is_test]

        if full_test_split_active:
            full_test_cam_extrinsics, full_test_cam_intrinsics, _ = _read_colmap_extrinsics_intrinsics(full_test_source_path)
            full_test_reading_dir = full_test_images if full_test_images else reading_dir
            full_test_cam_infos_unsorted = readColmapCameras(
                cam_extrinsics=full_test_cam_extrinsics,
                cam_intrinsics=full_test_cam_intrinsics,
                depths_params=None,
                images_folder=os.path.join(full_test_source_path, full_test_reading_dir),
                depths_folder="",
                test_cam_names_list=[],
            )
            full_test_cam_infos = sorted(full_test_cam_infos_unsorted.copy(), key=lambda x: x.image_name)
            train_cam_infos = [c._replace(is_test=False) for c in train_cam_infos]
            test_cam_infos = _build_full_source_test_split(train_cam_infos, full_test_cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        if use_auto_external_test:
            raise FileNotFoundError(
                "[SPLIT-READER][ERROR] train/sparse/0/points3D.ply is required "
                f"for external_test_source_path runs: {ply_path}"
            )
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False,
                           split_manifest=split_manifest)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True,
                           split_manifest={})
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
