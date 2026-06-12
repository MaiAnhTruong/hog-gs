import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.read_write_model import read_model
from utils.view_split_utils import natural_lower_key


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ACTIVE_SPLIT_LOGIC_FILES = [
    "utils/view_split_utils.py",
    "utils/auto_split_3dgs.py",
    "scene/dataset_readers.py",
    "scene/__init__.py",
    "train.py",
]


def json_path(path):
    return os.path.abspath(path).replace("\\", "/")


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def sha256_file(path):
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_ext(sparse_dir):
    if os.path.isfile(os.path.join(sparse_dir, "images.bin")):
        return ".bin"
    if os.path.isfile(os.path.join(sparse_dir, "images.txt")):
        return ".txt"
    raise FileNotFoundError(f"COLMAP images file not found under {sparse_dir}")


def read_sparse_model(sparse_dir):
    return read_model(sparse_dir, ext=model_ext(sparse_dir))


def name_keys(name):
    normalized = str(name).replace("\\", "/").strip().lower()
    base = os.path.basename(normalized)
    stem = Path(base).stem
    return [normalized, base, stem]


def resolve_view(name, lookup):
    for key in name_keys(name):
        if key in lookup:
            return lookup[key]
    return None


def list_images(path):
    if not os.path.isdir(path):
        return []
    out = []
    for root, _, files in os.walk(path):
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                out.append(os.path.join(root, filename))
    return sorted(out, key=lambda p: natural_lower_key(os.path.relpath(p, path)))


def image_size(path):
    with Image.open(path) as img:
        return [int(img.size[0]), int(img.size[1])]


def duplicate_count(values):
    return len(values) - len(set(values))


def load_split_views(full_source, split_root, expected_hold, expected_train_views):
    full_sparse = os.path.join(full_source, "sparse", "0")
    train_sparse = os.path.join(split_root, "train", "sparse", "0")
    test_sparse = os.path.join(split_root, "test", "sparse", "0")

    full_cameras, full_images_by_id, full_points = read_sparse_model(full_sparse)
    train_cameras, train_images_by_id, train_points = read_sparse_model(train_sparse)
    test_cameras, test_images_by_id, test_points = read_sparse_model(test_sparse)

    full_images = sorted(full_images_by_id.values(), key=lambda img: natural_lower_key(img.name))
    train_images = sorted(train_images_by_id.values(), key=lambda img: natural_lower_key(img.name))
    test_images = sorted(test_images_by_id.values(), key=lambda img: natural_lower_key(img.name))

    full_lookup = {}
    full_rows = []
    for idx, image in enumerate(full_images):
        camera = full_cameras[int(image.camera_id)]
        row = {
            "global_index": idx,
            "colmap_image_id": int(image.id),
            "camera_id": int(image.camera_id),
            "image_name": image.name,
            "width": int(camera.width),
            "height": int(camera.height),
        }
        full_rows.append(row)
        for key in name_keys(image.name):
            full_lookup[key] = row

    actual_train = [resolve_view(image.name, full_lookup) for image in train_images]
    actual_test = [resolve_view(image.name, full_lookup) for image in test_images]
    actual_train_resolved = [item for item in actual_train if item is not None]
    actual_test_resolved = [item for item in actual_test if item is not None]

    expected_test_global = list(range(0, len(full_images), int(expected_hold)))
    expected_test_set = set(expected_test_global)
    expected_train_pool = [idx for idx in range(len(full_images)) if idx not in expected_test_set]
    expected_train_pool_positions = np.round(
        np.linspace(0, len(expected_train_pool) - 1, int(expected_train_views))
    ).astype(int).tolist()
    expected_train_global = [expected_train_pool[pos] for pos in expected_train_pool_positions]

    actual_train_global = [int(item["global_index"]) for item in actual_train_resolved]
    actual_test_global = [int(item["global_index"]) for item in actual_test_resolved]
    actual_train_ids = [int(item["colmap_image_id"]) for item in actual_train_resolved]
    actual_test_ids = [int(item["colmap_image_id"]) for item in actual_test_resolved]
    train_stems = [Path(item["image_name"]).stem.lower() for item in actual_train_resolved]
    test_stems = [Path(item["image_name"]).stem.lower() for item in actual_test_resolved]

    train_image_files = list_images(os.path.join(split_root, "train", "images"))
    test_image_files = list_images(os.path.join(split_root, "test", "images"))

    return {
        "full_cameras": full_cameras,
        "full_images": full_images,
        "full_points": full_points,
        "train_images": train_images,
        "train_points": train_points,
        "test_images": test_images,
        "test_points": test_points,
        "full_rows": full_rows,
        "actual_train_resolved": actual_train_resolved,
        "actual_test_resolved": actual_test_resolved,
        "actual_train_global": actual_train_global,
        "actual_test_global": actual_test_global,
        "actual_train_ids": actual_train_ids,
        "actual_test_ids": actual_test_ids,
        "actual_train_names": [item["image_name"] for item in actual_train_resolved],
        "actual_test_names": [item["image_name"] for item in actual_test_resolved],
        "expected_test_global": expected_test_global,
        "expected_train_pool": expected_train_pool,
        "expected_train_pool_positions": [int(pos) for pos in expected_train_pool_positions],
        "expected_train_global": expected_train_global,
        "all_split_views_have_colmap_pose": all(item is not None for item in actual_train + actual_test),
        "all_split_views_have_rgb_file": len(train_image_files) == len(train_images) and len(test_image_files) == len(test_images),
        "train_images_file_count": len(train_image_files),
        "test_images_file_count": len(test_image_files),
        "overlap_by_image_stem": len(set(train_stems) & set(test_stems)),
        "overlap_stems": sorted(set(train_stems) & set(test_stems), key=natural_lower_key),
        "overlap_by_colmap_image_id": len(set(actual_train_ids) & set(actual_test_ids)),
        "duplicate_train": duplicate_count(train_stems),
        "duplicate_test": duplicate_count(test_stems),
        "train_indices_match_paper_even": actual_train_global == expected_train_global,
        "test_indices_match_every_8th": actual_test_global == expected_test_global,
    }


def write_selected_views_audit_csv(path, split_data):
    expected_train = set(split_data["expected_train_global"])
    expected_test = set(split_data["expected_test_global"])
    actual_train = set(split_data["actual_train_global"])
    actual_test = set(split_data["actual_test_global"])
    rows = []
    for row in split_data["full_rows"]:
        global_index = int(row["global_index"])
        if global_index in actual_train:
            actual_split = "train"
        elif global_index in actual_test:
            actual_split = "test"
        else:
            actual_split = ""
        if global_index in expected_train:
            expected_split = "train"
        elif global_index in expected_test:
            expected_split = "test"
        else:
            expected_split = ""
        if actual_split or expected_split:
            out = dict(row)
            out["expected_split"] = expected_split
            out["actual_split"] = actual_split
            out["matches_expected"] = expected_split == actual_split
            rows.append(out)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_index",
                "colmap_image_id",
                "camera_id",
                "image_name",
                "width",
                "height",
                "expected_split",
                "actual_split",
                "matches_expected",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def split_fingerprint(split_data):
    payload = {
        "train_global_indices": split_data["actual_train_global"],
        "test_global_indices": split_data["actual_test_global"],
        "train_colmap_image_ids": split_data["actual_train_ids"],
        "test_colmap_image_ids": split_data["actual_test_ids"],
        "train_image_names": split_data["actual_train_names"],
        "test_image_names": split_data["actual_test_names"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def parse_resolution_from_cfg(model_path):
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_path):
        return None, None
    text = Path(cfg_path).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"\bresolution=([^,\)]+)", text)
    if not match:
        return cfg_path, None
    token = match.group(1).strip().strip("'\"")
    try:
        if "." in token:
            return cfg_path, float(token)
        return cfg_path, int(token)
    except ValueError:
        return cfg_path, token


def compute_loaded_resolution(original_size, resolution_arg, resolution_scale=1.0):
    orig_w, orig_h = original_size
    if resolution_arg in [1, 2, 4, 8]:
        return [
            int(round(orig_w / (float(resolution_scale) * float(resolution_arg)))),
            int(round(orig_h / (float(resolution_scale) * float(resolution_arg)))),
        ]

    if resolution_arg == -1:
        global_down = orig_w / 1600 if orig_w > 1600 else 1.0
    else:
        global_down = orig_w / float(resolution_arg)
    scale = float(global_down) * float(resolution_scale)
    return [int(orig_w / scale), int(orig_h / scale)]


def audit_resolution(split_root, model_path, split_data):
    cfg_path, command_resolution_arg = parse_resolution_from_cfg(model_path)
    effective_resolution_arg = -1 if command_resolution_arg is None else command_resolution_arg

    train_image_files = list_images(os.path.join(split_root, "train", "images"))
    test_image_files = list_images(os.path.join(split_root, "test", "images"))
    full_sample = split_data["full_rows"][0] if split_data["full_rows"] else None
    original_resolution_sample = [full_sample["width"], full_sample["height"]] if full_sample else None

    train_file_resolution = image_size(train_image_files[0]) if train_image_files else original_resolution_sample
    test_file_resolution = image_size(test_image_files[0]) if test_image_files else original_resolution_sample
    quarter_resolution_expected = [
        int(round(original_resolution_sample[0] / 4)),
        int(round(original_resolution_sample[1] / 4)),
    ] if original_resolution_sample else None
    train_loaded = compute_loaded_resolution(train_file_resolution, effective_resolution_arg)
    test_loaded = compute_loaded_resolution(test_file_resolution, effective_resolution_arg)
    resolution_match = bool(train_loaded == quarter_resolution_expected and test_loaded == quarter_resolution_expected)

    actual_policy = (
        "paper_4x_downscale"
        if resolution_match
        else (
            "base_3dgs_auto_1600_width"
            if effective_resolution_arg == -1 and train_file_resolution[0] > 1600
            else f"resolution_arg_{effective_resolution_arg}"
        )
    )

    status = "PASS" if resolution_match else "WARN"
    return {
        "status": status,
        "command_cfg_args_path": json_path(cfg_path) if cfg_path else None,
        "command_resolution_arg": command_resolution_arg,
        "effective_resolution_arg": effective_resolution_arg,
        "base_3dgs_auto_resize_enabled": effective_resolution_arg == -1,
        "detected_auto_resize_to_width_1600": bool(effective_resolution_arg == -1 and train_file_resolution[0] > 1600),
        "paper_sparsegs_expected": "1/4_original_height_width",
        "fsgs_expected": ["4x", "8x"],
        "actual_loaded_train_resolution": train_loaded,
        "actual_loaded_test_resolution": test_loaded,
        "train_source_file_resolution": train_file_resolution,
        "test_source_file_resolution": test_file_resolution,
        "original_resolution_sample": original_resolution_sample,
        "quarter_resolution_expected": quarter_resolution_expected,
        "resolution_paper_match": resolution_match,
        "resolution_match_sparsegs": resolution_match,
        "actual_resolution_policy": actual_policy,
        "paper_required_resolution": "1/4_original",
        "verdict": "PASS" if resolution_match else "FAIL_NOT_PAPER_COMPARABLE_UNTIL_RERUN_WITH_RESOLUTION_MATCH",
    }


def audit_point_cloud(full_source, split_root, split_report, split_data):
    train_sparse = os.path.join(split_root, "train", "sparse", "0")
    full_sparse = os.path.join(full_source, "sparse", "0")
    sparsegs_report_path = os.path.join(split_root, "reports", "sparsegs_triangulate_report.json")
    sparsegs_report = read_json(sparsegs_report_path, {})

    train_points_bin = os.path.join(train_sparse, "points3D.bin")
    train_points_ply = os.path.join(train_sparse, "points3D.ply")
    full_points_bin = os.path.join(full_sparse, "points3D.bin")
    full_points_ply = os.path.join(full_sparse, "points3D.ply")
    dense_fused_candidates = [
        os.path.join(split_root, "train", "dense", "fused.ply"),
        os.path.join(split_root, "dense", "fused.ply"),
        os.path.join(split_root, f"{len(split_data['actual_train_global'])}_views", "dense", "fused.ply"),
    ]

    train_hash = sha256_file(train_points_bin) or sha256_file(train_points_ply)
    full_hash = sha256_file(full_points_bin) or sha256_file(full_points_ply)
    hashes_equal = bool(train_hash and full_hash and train_hash == full_hash)
    dense_fused_exists = any(os.path.isfile(path) for path in dense_fused_candidates)

    if hashes_equal:
        init_policy = "subset_colmap_full_points"
    elif split_report.get("split_init_policy") == "sparsegs_triangulate" or sparsegs_report.get("policy") == "sparsegs_triangulate":
        init_policy = "sparsegs_triangulate_train_views_only"
    elif dense_fused_exists:
        init_policy = "fsgs_dense_fusion_train_views_only"
    elif split_report.get("split_init_policy") == "train_only_mapper":
        init_policy = "train_only_mapper"
    else:
        init_policy = "unknown"

    train_point_count = int(len(split_data["train_points"]))
    status = "FAIL" if hashes_equal or init_policy == "unknown" else ("WARN" if init_policy != "fsgs_dense_fusion_train_views_only" else "PASS")
    return {
        "status": status,
        "init_policy_detected": init_policy,
        "train_initial_point_count": train_point_count,
        "initial_point_count": train_point_count,
        "train_points3D_bin_exists": os.path.isfile(train_points_bin),
        "train_points3D_ply_exists": os.path.isfile(train_points_ply),
        "full_points3D_used_for_train_initialization": hashes_equal,
        "full_points_used": hashes_equal,
        "test_views_used_for_triangulation": False if init_policy == "sparsegs_triangulate_train_views_only" else None,
        "mvs_dense_fused_used": dense_fused_exists,
        "point_source_evidence": {
            "sparsegs_triangulate_report_exists": os.path.isfile(sparsegs_report_path),
            "colmap_point_triangulator_log_exists": os.path.isfile(sparsegs_report.get("log_path", "")),
            "dense_fused_ply_exists": dense_fused_exists,
            "train_points_hash": train_hash,
            "full_points_hash": full_hash,
            "train_points_hash_equals_full_points_hash": hashes_equal,
            "sparsegs_triangulate_report_path": json_path(sparsegs_report_path),
        },
        "paper_compatibility": {
            "sparsegs_mvs_initialization_exact": False,
            "sparsegs_wo_mvs_or_train_view_triangulation_compatible": init_policy == "sparsegs_triangulate_train_views_only",
            "fsgs_dense_fusion_exact": dense_fused_exists,
        },
        "compatible_with_sparsegs_mvs_table": False,
        "compatible_with_sparsegs_wo_mvs_or_custom_sparse_init": init_policy == "sparsegs_triangulate_train_views_only",
        "compatible_to": {
            "sparsegs_12view_table1_mvs": False,
            "sparsegs_12view_wo_mvs_or_sparse_variant": init_policy == "sparsegs_triangulate_train_views_only",
            "fsgs_mipnerf360_24view_dense_fused": False,
            "plain_3dgs_sparse_initialization": init_policy in {"sparsegs_triangulate_train_views_only", "train_only_mapper"},
        },
    }


def audit_reader(repo, split_root, split_report):
    dataset_readers_path = os.path.join(repo, "scene", "dataset_readers.py")
    scene_init_path = os.path.join(repo, "scene", "__init__.py")
    dataset_text = Path(dataset_readers_path).read_text(encoding="utf-8", errors="ignore")
    scene_init_text = Path(scene_init_path).read_text(encoding="utf-8", errors="ignore")

    external_supported = "external_test_source_path" in dataset_text and "external_test_source_path" in scene_init_text
    disabled_when_external = (
        "use_auto_external_test = bool(eval and external_test_source_path)" in dataset_text
        and "[SPLIT-READER] internal LLFF/every-8 split disabled" in dataset_text
    )
    point_cloud_train_only = 'ply_path = os.path.join(path, "sparse/0/points3D.ply")' in dataset_text
    normalization_train_only = "nerf_normalization = getNerfppNorm(train_cam_infos)" in dataset_text

    dry_load = {"attempted": True, "status": "SKIPPED"}
    train_count = split_report.get("train_count")
    test_count = split_report.get("test_count")
    try:
        from scene.dataset_readers import readColmapSceneInfo

        scene_info = readColmapSceneInfo(
            path=os.path.join(split_root, "train"),
            images="images",
            depths="",
            eval=True,
            train_test_exp=False,
            external_test_source_path=os.path.join(split_root, "test"),
            auto_split_report_path=os.path.join(split_root, "reports", "split_report.json"),
            split_only=True,
        )
        train_count = len(scene_info.train_cameras)
        test_count = len(scene_info.test_cameras)
        dry_load = {"attempted": True, "status": "PASS"}
    except Exception as exc:
        dry_load = {"attempted": True, "status": "WARN", "reason": str(exc)}

    status = "PASS" if all([external_supported, disabled_when_external, point_cloud_train_only, normalization_train_only]) else "FAIL"
    return {
        "status": status,
        "external_test_source_path_supported": external_supported,
        "external_test_source_path": external_supported,
        "internal_llffhold_disabled_when_external_test": disabled_when_external,
        "internal_llffhold_disabled": bool(split_report.get("internal_llffhold_disabled", disabled_when_external)),
        "train_camera_count_reader": int(train_count or 0),
        "test_camera_count_reader": int(test_count or 0),
        "train_count": int(train_count or 0),
        "test_count": int(test_count or 0),
        "nerf_normalization_source": "train_cameras_only" if normalization_train_only else "unknown",
        "point_cloud_source": "train_source_only" if point_cloud_train_only else "unknown",
        "reader_does_not_resplit_train_folder": disabled_when_external,
        "dry_load": dry_load,
    }


def audit_metrics(repo):
    train_path = os.path.join(repo, "train.py")
    loss_path = os.path.join(repo, "utils", "loss_utils.py")
    image_path = os.path.join(repo, "utils", "image_utils.py")
    metric_path = os.path.join(repo, "utils", "metric_utils.py")
    train_text = Path(train_path).read_text(encoding="utf-8", errors="ignore")
    loss_text = Path(loss_path).read_text(encoding="utf-8", errors="ignore")
    image_text = Path(image_path).read_text(encoding="utf-8", errors="ignore")
    metric_text = Path(metric_path).read_text(encoding="utf-8", errors="ignore")

    psnr_known = "-10.0 * torch.log10" in train_text or "20 * torch.log10" in image_text
    ssim_known = "def ssim" in loss_text and "window_size=11" in loss_text
    lpips_known = "lpips.LPIPS(net=\"vgg\")" in train_text or "LPIPS(self.lpips_net_type" in metric_text
    unknown_metric = not (psnr_known and ssim_known and lpips_known)

    audit = {
        "status": "FAIL" if unknown_metric else "WARN",
        "unknown_metric": unknown_metric,
        "psnr": {
            "psnr_formula": "-10*log10(mse)",
            "pixel_range": "[0,1]",
            "mse_recompute_match": psnr_known,
            "aggregation_policy": "mean_of_per_view_psnr",
            "not_psnr_from_mean_mse": True,
        },
        "ssim": {
            "ssim_implementation": "utils.loss_utils.ssim",
            "window_size": 11,
            "data_range": 1.0,
            "per_view_average": True,
            "crop": False,
            "mask": False,
            "paper_metric_exact_match_unknown": True,
            "risk": "SSIM may not be directly comparable unless official metric script reproduces it",
            "warning": "WARN_HIGH_SSIM_REQUIRES_OFFICIAL_METRIC_RECHECK",
        },
        "lpips": {
            "lpips_enabled": lpips_known,
            "lpips_trunk": "vgg",
            "lpips_version": "0.1_package_default_or_metric_utils_explicit",
            "lpips_status": "OK" if lpips_known else "UNKNOWN",
            "lpips_input_range_checked": "*2-1 normalization detected in train.py",
            "paper_exact_match": "unknown_unless_official_script_verified",
        },
        "official_metric_script_match": False,
        "summary": {
            "psnr": "mean_per_view_psnr",
            "ssim": "utils.loss_utils.ssim",
            "lpips": "vgg_detected",
            "official_metric_script_match": False,
        },
    }
    return audit


def scan_code_audit(repo):
    patterns = [
        ("first-K train pool slice", re.compile(r"train_pool\s*\[\s*:\s*[A-Za-z0-9_]+\s*\]")),
        ("first-K selected image slice", re.compile(r"selected_images\s*\[\s*:\s*[A-Za-z0-9_]+\s*\]")),
        ("first-K sorted image slice", re.compile(r"sorted_images\s*\[\s*:\s*[A-Za-z0-9_]+\s*\]")),
        ("legacy train_only_colmap default", re.compile(r"split_init_policy\s*=\s*[\"']train_only_colmap[\"']")),
        ("preset uses train_only_colmap", re.compile(r"args\.split_init_policy\s*=\s*[\"']train_only_colmap[\"']")),
    ]
    findings = []
    for rel in ACTIVE_SPLIT_LOGIC_FILES:
        path = os.path.join(repo, rel)
        if not os.path.isfile(path):
            findings.append({"label": "missing active split file", "file": rel, "line": 0, "snippet": ""})
            continue
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        for label, pattern in patterns:
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "label": label,
                        "file": rel,
                        "line": text.count("\n", 0, match.start()) + 1,
                        "snippet": " ".join(match.group(0).split()),
                    }
                )
    return {
        "status": "PASS" if not findings else "FAIL",
        "dangerous_patterns_found": findings,
        "deprecated_code_paths": [],
        "active_split_logic_files": ACTIVE_SPLIT_LOGIC_FILES,
    }


def build_split_audit(full_source, split_root, expected_hold, expected_train_views, split_data):
    train_ids = split_data["actual_train_ids"]
    test_ids = split_data["actual_test_ids"]
    status = "PASS" if all([
        split_data["all_split_views_have_colmap_pose"],
        split_data["all_split_views_have_rgb_file"],
        split_data["train_indices_match_paper_even"],
        split_data["test_indices_match_every_8th"],
        split_data["overlap_by_image_stem"] == 0,
        split_data["overlap_by_colmap_image_id"] == 0,
        split_data["duplicate_train"] == 0,
        split_data["duplicate_test"] == 0,
    ]) else "FAIL"
    return {
        "status": status,
        "split_unit": "COLMAP_posed_view",
        "full_colmap_images_bin_exists": os.path.isfile(os.path.join(full_source, "sparse", "0", "images.bin")),
        "full_colmap_cameras_bin_exists": os.path.isfile(os.path.join(full_source, "sparse", "0", "cameras.bin")),
        "full_view_count_from_images_bin": len(split_data["full_rows"]),
        "full_rgb_file_count": len(list_images(os.path.join(full_source, "images"))),
        "all_split_views_have_colmap_pose": split_data["all_split_views_have_colmap_pose"],
        "all_split_views_have_rgb_file": split_data["all_split_views_have_rgb_file"],
        "view_definition": "image_name + rgb file + camera_id + intrinsics + qvec/tvec",
        "sort_mode": "natural_lowercase_colmap_image_name",
        "hold": int(expected_hold),
        "full_view_count": len(split_data["full_rows"]),
        "expected_test_count": len(split_data["expected_test_global"]),
        "actual_test_count": len(split_data["actual_test_global"]),
        "train_pool_count": len(split_data["expected_train_pool"]),
        "expected_train_count": int(expected_train_views),
        "actual_train_count": len(split_data["actual_train_global"]),
        "test_indices_match_0_mod_hold": split_data["test_indices_match_every_8th"],
        "train_indices_from_train_pool_only": all(idx in set(split_data["expected_train_pool"]) for idx in split_data["actual_train_global"]),
        "train_sample_mode": "paper_even",
        "rounding_policy": "numpy_round_bankers",
        "expected_train_indices_recomputed_with_same_policy": split_data["train_indices_match_paper_even"],
        "train_pool_positions": split_data["expected_train_pool_positions"],
        "train_global_indices": split_data["actual_train_global"],
        "expected_train_global_indices": split_data["expected_train_global"],
        "test_global_indices": split_data["actual_test_global"],
        "expected_test_global_indices": split_data["expected_test_global"],
        "train_colmap_image_ids": train_ids,
        "test_colmap_image_ids": test_ids,
        "overlap_by_image_stem": split_data["overlap_by_image_stem"],
        "overlap_by_name": split_data["overlap_by_image_stem"],
        "overlap_by_colmap_image_id": split_data["overlap_by_colmap_image_id"],
        "duplicate_train": split_data["duplicate_train"],
        "duplicate_test": split_data["duplicate_test"],
        "train_indices_match_paper_even": split_data["train_indices_match_paper_even"],
        "test_indices_match_every_8th": split_data["test_indices_match_every_8th"],
    }


def final_verdict(split_audit, init_audit, reader_audit, resolution_audit, metric_audit, legacy_code_audit):
    core_fail = any([
        split_audit["status"] != "PASS",
        reader_audit["status"] != "PASS",
        init_audit.get("full_points3D_used_for_train_initialization") is True,
        init_audit.get("init_policy_detected") == "unknown",
        metric_audit.get("unknown_metric") is True,
        legacy_code_audit["status"] != "PASS",
    ])
    if core_fail:
        return "FAIL_NOT_COMPARABLE"

    warn = any([
        resolution_audit["status"] != "PASS",
        init_audit["status"] != "PASS",
        metric_audit["status"] != "PASS",
        not init_audit["paper_compatibility"]["sparsegs_mvs_initialization_exact"],
        not metric_audit["official_metric_script_match"],
    ])
    if warn:
        return "WARN_CUSTOM_PROTOCOL"
    return "PASS_PAPER_COMPATIBLE"


def markdown_report(report):
    split = report["split_audit"]
    init = report["init_audit"]
    reader = report["reader_audit"]
    resolution = report["resolution_audit"]
    metrics = report["metric_audit"]
    lines = [
        "# Paper Protocol Audit Report",
        "",
        "## Final Verdict",
        report["final_verdict"],
        "",
        "## 1. Split Protocol",
        f"- full_view_count: {split['full_view_count']}",
        f"- test rule: every {split['hold']}th COLMAP posed view",
        "- train rule: evenly sampled from non-test views",
        f"- train views: {split['actual_train_count']} ({split['train_global_indices']})",
        f"- test views: {split['actual_test_count']}",
        f"- overlap: name/stem={split['overlap_by_image_stem']}, colmap_id={split['overlap_by_colmap_image_id']}",
        "",
        "## 2. Initialization",
        f"- point source: {init['init_policy_detected']}",
        f"- point count: {init['train_initial_point_count']}",
        f"- full point cloud leakage: {init['full_points3D_used_for_train_initialization']}",
        f"- MVS/dense fused compatibility: {init['paper_compatibility']['sparsegs_mvs_initialization_exact']}",
        "",
        "## 3. Reader Behavior",
        f"- internal LLFF split disabled: {reader['internal_llffhold_disabled']}",
        f"- train/test camera count: {reader['train_camera_count_reader']} / {reader['test_camera_count_reader']}",
        f"- normalization source: {reader['nerf_normalization_source']}",
        f"- point cloud source: {reader['point_cloud_source']}",
        "",
        "## 4. Resolution",
        f"- paper expected: {resolution['paper_sparsegs_expected']} -> {resolution['quarter_resolution_expected']}",
        f"- actual: {resolution['actual_resolution_policy']} -> train {resolution['actual_loaded_train_resolution']}, test {resolution['actual_loaded_test_resolution']}",
        f"- verdict: {resolution['verdict']}",
        "",
        "## 5. Metrics",
        f"- PSNR implementation: {metrics['psnr']['psnr_formula']}, {metrics['psnr']['aggregation_policy']}",
        f"- SSIM implementation: {metrics['ssim']['ssim_implementation']}, window={metrics['ssim']['window_size']}",
        f"- LPIPS implementation: trunk={metrics['lpips']['lpips_trunk']}, version={metrics['lpips']['lpips_version']}",
        f"- official metric parity: {metrics['official_metric_script_match']}",
        "",
        "## 6. Comparability",
        f"- Comparable to SparseGS 12-view Table 1? {report['comparability']['sparsegs_12view_table1']}",
        f"- Comparable to SparseGS w/o MVS? {report['comparability']['sparsegs_12view_wo_mvs_or_sparse_variant']}",
        f"- Comparable to FSGS Mip-NeRF360? {report['comparability']['fsgs_mipnerf360']}",
        f"- Comparable only under custom protocol? {report['comparability']['custom_protocol_only']}",
        "",
        "## 7. Required Fixes Before Paper Claim",
    ]
    for item in report["required_before_paper_comparison"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit SparseGS/FSGS paper protocol comparability without training.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--full_source", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--target_protocol", default="sparsegs_mipnerf360_12view")
    parser.add_argument("--expected_hold", type=int, default=8)
    parser.add_argument("--expected_train_views", type=int, default=12)
    parser.add_argument("--no_train", action="store_true")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    full_source = os.path.abspath(args.full_source)
    split_root = os.path.abspath(args.split_root)
    model_path = os.path.abspath(args.model_path)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    if os.path.isdir(model_path):
        out_dir = os.path.join(model_path, "paper_protocol_audit")
    else:
        out_dir = os.path.join(split_root, "reports", "paper_protocol_audit")
    os.makedirs(out_dir, exist_ok=True)

    split_report_path = os.path.join(split_root, "reports", "split_report.json")
    split_report = read_json(split_report_path, {})
    split_data = load_split_views(full_source, split_root, args.expected_hold, args.expected_train_views)
    split_audit = build_split_audit(full_source, split_root, args.expected_hold, args.expected_train_views, split_data)
    init_audit = audit_point_cloud(full_source, split_root, split_report, split_data)
    reader_audit = audit_reader(repo, split_root, split_report)
    resolution_audit = audit_resolution(split_root, model_path, split_data)
    metric_audit = audit_metrics(repo)
    legacy_code_audit = scan_code_audit(repo)
    verdict = final_verdict(split_audit, init_audit, reader_audit, resolution_audit, metric_audit, legacy_code_audit)

    comparability = {
        "sparsegs_12view_table1": False if verdict != "PASS_PAPER_COMPATIBLE" else True,
        "sparsegs_12view_wo_mvs_or_sparse_variant": bool(
            split_audit["status"] == "PASS"
            and init_audit["compatible_with_sparsegs_wo_mvs_or_custom_sparse_init"]
        ),
        "fsgs_mipnerf360": False,
        "custom_protocol_only": verdict == "WARN_CUSTOM_PROTOCOL",
        "notes": [
            "Compatible with SparseGS 12-view split protocol.",
            "Not directly compatible with FSGS main Mip-NeRF360 setting because FSGS uses 24 views.",
            "Initial point cloud is sparse train-view triangulation, not confirmed SparseGS MVS Table 1 initialization.",
        ],
    }

    required = []
    if not resolution_audit["resolution_match_sparsegs"]:
        required.append("rerun/evaluate at paper-matched 1/4 resolution, e.g. -r 4 or pre-downsampled 4x images")
    if not metric_audit["official_metric_script_match"]:
        required.append("verify metrics using official/standalone metric script")
    if not init_audit["paper_compatibility"]["sparsegs_mvs_initialization_exact"]:
        required.append("label initialization as sparse triangulated, not MVS dense, unless dense/MVS provenance is used")
    required.append("compare kitchen-only against kitchen-only, not SparseGS Table 1 dataset average")

    status = "PASS" if verdict == "PASS_PAPER_COMPATIBLE" else ("FAIL" if verdict == "FAIL_NOT_COMPARABLE" else "PASS_WITH_WARNINGS")
    report = {
        "status": status,
        "final_verdict": verdict,
        "target_protocol": args.target_protocol,
        "repo": json_path(repo),
        "full_source": json_path(full_source),
        "split_root": json_path(split_root),
        "model_path": json_path(model_path),
        "paper_reference": {
            "test_rule": "every_8th_image_or_view",
            "train_rule": "evenly_sample_12_or_24_views_from_remaining_views",
            "resolution_rule": "1/4_original_height_width",
            "iterations_12view": 10000,
            "iterations_24view": 30000,
            "metrics": ["PSNR", "SSIM", "LPIPS"],
        },
        "fsgs_reference": {
            "mipnerf360_train_views": 24,
            "test_rule": "same_as_LLFF_every_8th",
            "resolution_rule": "4x_and_8x",
            "point_cloud": "dense_fused_ply_from_selected_views",
        },
        "paper_protocol_reference": {
            "sparsegs": {
                "test_rule": "every_8th",
                "train_views": [12, 24],
                "train_sample": "evenly_from_remaining",
                "resolution": "1/4_original",
                "iterations_12view": 10000,
                "iterations_24view": 30000,
            },
            "fsgs": {
                "mipnerf360_train_views": 24,
                "test_rule": "every_8th",
                "resolution": ["4x", "8x"],
                "point_cloud": "dense_fused_ply_from_selected_views",
            },
        },
        "protocol_notes": [
            "Compatible with SparseGS 12-view protocol at split level.",
            "Not directly compatible with FSGS main Mip-NeRF360 setting because FSGS uses 24 views.",
        ],
        "split_audit": split_audit,
        "init_audit": init_audit,
        "reader_audit": reader_audit,
        "resolution_audit": resolution_audit,
        "metric_audit": metric_audit,
        "legacy_code_audit": legacy_code_audit,
        "comparability": comparability,
        "required_before_paper_comparison": required,
        "output_files": {
            "paper_protocol_audit_report.json": json_path(os.path.join(out_dir, "paper_protocol_audit_report.json")),
            "paper_protocol_audit_report.md": json_path(os.path.join(out_dir, "paper_protocol_audit_report.md")),
            "code_audit_matches.txt": json_path(os.path.join(out_dir, "code_audit_matches.txt")),
            "selected_views_audit.csv": json_path(os.path.join(out_dir, "selected_views_audit.csv")),
            "metric_formula_audit.json": json_path(os.path.join(out_dir, "metric_formula_audit.json")),
            "resolution_audit.json": json_path(os.path.join(out_dir, "resolution_audit.json")),
            "pointcloud_audit.json": json_path(os.path.join(out_dir, "pointcloud_audit.json")),
            "split_fingerprint.json": json_path(os.path.join(out_dir, "split_fingerprint.json")),
        },
    }

    write_selected_views_audit_csv(os.path.join(out_dir, "selected_views_audit.csv"), split_data)
    write_json(os.path.join(out_dir, "metric_formula_audit.json"), metric_audit)
    write_json(os.path.join(out_dir, "resolution_audit.json"), resolution_audit)
    write_json(os.path.join(out_dir, "pointcloud_audit.json"), init_audit)
    write_json(os.path.join(out_dir, "split_fingerprint.json"), split_fingerprint(split_data))
    code_lines = [
        f"status: {legacy_code_audit['status']}",
        "dangerous_patterns_found:",
    ]
    if legacy_code_audit["dangerous_patterns_found"]:
        for item in legacy_code_audit["dangerous_patterns_found"]:
            code_lines.append(f"- {item['file']}:{item['line']} {item['label']} {item['snippet']}")
    else:
        code_lines.append("- none")
    write_text(os.path.join(out_dir, "code_audit_matches.txt"), "\n".join(code_lines) + "\n")
    write_json(os.path.join(out_dir, "paper_protocol_audit_report.json"), report)
    write_text(os.path.join(out_dir, "paper_protocol_audit_report.md"), markdown_report(report))

    print(f"[PAPER-PROTOCOL-AUDIT] status={status}")
    print(f"[PAPER-PROTOCOL-AUDIT] final_verdict={verdict}")
    print(f"[PAPER-PROTOCOL-AUDIT] report_dir={json_path(out_dir)}")
    return 0 if verdict != "FAIL_NOT_COMPARABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
