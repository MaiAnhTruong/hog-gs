import os
import shutil
from typing import Dict, Iterable, Sequence

import numpy as np

from utils.read_write_model import (
    Image,
    Point3D,
    write_cameras_binary as _write_cameras_binary,
    write_images_binary as _write_images_binary,
    write_model,
    write_points3D_binary as _write_points3D_binary,
)


def write_cameras_binary(cameras: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return _write_cameras_binary(cameras, path)


def write_images_binary(images: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return _write_images_binary(images, path)


def write_points3D_binary(points3D: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return _write_points3D_binary(points3D, path)


def clone_pose_only_image(image):
    return Image(
        id=int(image.id),
        qvec=np.asarray(image.qvec, dtype=np.float64).copy(),
        tvec=np.asarray(image.tvec, dtype=np.float64).copy(),
        camera_id=int(image.camera_id),
        name=image.name,
        xys=np.empty((0, 2), dtype=np.float64),
        point3D_ids=np.empty((0,), dtype=np.int64),
    )


def clone_point_without_track(point):
    return Point3D(
        id=int(point.id),
        xyz=np.asarray(point.xyz, dtype=np.float64).copy(),
        rgb=np.asarray(point.rgb, dtype=np.uint8).copy(),
        error=float(np.asarray(point.error).item()),
        image_ids=np.empty((0,), dtype=np.int32),
        point2D_idxs=np.empty((0,), dtype=np.int32),
    )


def subset_cameras_for_images(cameras: dict, images: Iterable) -> Dict[int, object]:
    camera_ids = sorted({int(img.camera_id) for img in images})
    missing = [camera_id for camera_id in camera_ids if camera_id not in cameras]
    if missing:
        raise ValueError(f"missing camera ids for sparse model: {missing}")
    return {camera_id: cameras[camera_id] for camera_id in camera_ids}


def images_for_views(full_images: dict, views: Sequence) -> Dict[int, Image]:
    images = {}
    for view in views:
        image_id = int(view.colmap_image_id)
        if image_id not in full_images:
            raise ValueError(f"selected view missing from full COLMAP images: image_id={image_id}")
        images[image_id] = clone_pose_only_image(full_images[image_id])
    return images


def write_points3D_ply(points_xyz, points_rgb, ply_path: str):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    points_rgb = np.asarray(points_rgb, dtype=np.uint8)
    if points_xyz.size == 0:
        points_xyz = np.empty((0, 3), dtype=np.float64)
    if points_rgb.size == 0:
        points_rgb = np.empty((0, 3), dtype=np.uint8)
    if points_xyz.shape[0] != points_rgb.shape[0]:
        raise ValueError(
            f"points xyz/rgb count mismatch: xyz={points_xyz.shape}, rgb={points_rgb.shape}"
        )

    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    with open(ply_path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(points_xyz, points_rgb):
            f.write(
                f"{float(xyz[0])} {float(xyz[1])} {float(xyz[2])} "
                f"0 0 0 {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def points3d_dict_to_arrays(points3D: dict):
    if not points3D:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    points = [points3D[k] for k in sorted(points3D)]
    xyz = np.asarray([p.xyz for p in points], dtype=np.float64)
    rgb = np.asarray([p.rgb for p in points], dtype=np.uint8)
    return xyz, rgb


def arrays_to_trackless_points3d(points_xyz, points_rgb, start_id: int = 1):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    points_rgb = np.asarray(points_rgb, dtype=np.uint8)
    out = {}
    for idx, (xyz, rgb) in enumerate(zip(points_xyz, points_rgb), start=start_id):
        out[int(idx)] = Point3D(
            id=int(idx),
            xyz=np.asarray(xyz, dtype=np.float64),
            rgb=np.asarray(rgb, dtype=np.uint8),
            error=0.0,
            image_ids=np.empty((0,), dtype=np.int32),
            point2D_idxs=np.empty((0,), dtype=np.int32),
        )
    return out


def write_pose_only_sparse_model(output_sparse0: str, cameras_subset: dict, images_subset: dict):
    os.makedirs(output_sparse0, exist_ok=True)
    write_model(cameras_subset, images_subset, {}, output_sparse0, ext=".bin")
    write_points3D_ply(
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.uint8),
        os.path.join(output_sparse0, "points3D.ply"),
    )
    return {
        "camera_count": len(cameras_subset),
        "image_count": len(images_subset),
        "point_count": 0,
    }


def write_train_sparse_model(output_sparse0: str, cameras_subset: dict, images_subset: dict, points3d_aligned: dict):
    os.makedirs(output_sparse0, exist_ok=True)
    points_out = {int(pid): clone_point_without_track(point) for pid, point in points3d_aligned.items()}
    write_model(cameras_subset, images_subset, points_out, output_sparse0, ext=".bin")
    xyz, rgb = points3d_dict_to_arrays(points_out)
    write_points3D_ply(xyz, rgb, os.path.join(output_sparse0, "points3D.ply"))
    return {
        "camera_count": len(cameras_subset),
        "image_count": len(images_subset),
        "point_count": len(points_out),
    }


def materialize_image(src: str, dst: str, mode: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"unknown split_copy_mode: {mode}")


def copy_view_images(views: Sequence, output_source_path: str, copy_mode: str):
    images_dir = os.path.join(output_source_path, "images")
    for view in views:
        if not view.image_file_exists:
            raise FileNotFoundError(f"selected RGB image does not exist: {view.image_path}")
        rel_parts = str(view.image_name).replace("\\", "/").split("/")
        dst = os.path.join(images_dir, *rel_parts)
        materialize_image(view.image_path, dst, copy_mode)
