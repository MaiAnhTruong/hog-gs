import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.auto_split_3dgs import validate_split_report
from utils.read_write_model import read_model
from utils.view_split_utils import natural_lower_key


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _count_images(path):
    if not os.path.isdir(path):
        return 0
    count = 0
    for _, _, files in os.walk(path):
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                count += 1
    return count


def _read_model_images(sparse_dir):
    if os.path.isfile(os.path.join(sparse_dir, "images.bin")):
        _, images, _ = read_model(sparse_dir, ext=".bin")
    else:
        _, images, _ = read_model(sparse_dir, ext=".txt")
    return sorted(images.values(), key=lambda img: natural_lower_key(img.name))


def _name_keys(name):
    normalized = str(name).replace("\\", "/").strip().lower()
    base = os.path.basename(normalized)
    stem = Path(base).stem
    return [normalized, base, stem]


def _resolve_full_view(name, full_lookup):
    for key in _name_keys(name):
        if key in full_lookup:
            return full_lookup[key]
    return None


def _recompute_exact_split(full_source, split_root, expected_hold, expected_train_views):
    full_sparse = os.path.join(full_source, "sparse", "0")
    train_sparse = os.path.join(split_root, "train", "sparse", "0")
    test_sparse = os.path.join(split_root, "test", "sparse", "0")

    full_images = _read_model_images(full_sparse)
    train_images = _read_model_images(train_sparse)
    test_images = _read_model_images(test_sparse)

    full_lookup = {}
    for idx, image in enumerate(full_images):
        for key in _name_keys(image.name):
            full_lookup[key] = {
                "global_index": idx,
                "colmap_image_id": int(image.id),
                "image_name": image.name,
            }

    actual_train = [_resolve_full_view(image.name, full_lookup) for image in train_images]
    actual_test = [_resolve_full_view(image.name, full_lookup) for image in test_images]
    expected_test_global = list(range(0, len(full_images), int(expected_hold)))
    train_pool_global = [idx for idx in range(len(full_images)) if idx not in set(expected_test_global)]
    expected_train_pool_positions = np.round(
        np.linspace(0, len(train_pool_global) - 1, int(expected_train_views))
    ).astype(int).tolist()
    expected_train_global = [train_pool_global[pos] for pos in expected_train_pool_positions]

    actual_train_global = [item["global_index"] for item in actual_train if item is not None]
    actual_test_global = [item["global_index"] for item in actual_test if item is not None]
    actual_train_ids = [item["colmap_image_id"] for item in actual_train if item is not None]
    actual_test_ids = [item["colmap_image_id"] for item in actual_test if item is not None]

    return {
        "full_view_count": len(full_images),
        "train_pool_count": len(train_pool_global),
        "expected_test_global": expected_test_global,
        "expected_train_pool_positions": [int(pos) for pos in expected_train_pool_positions],
        "expected_train_global": expected_train_global,
        "actual_train_global": actual_train_global,
        "actual_test_global": actual_test_global,
        "actual_train_colmap_ids": actual_train_ids,
        "actual_test_colmap_ids": actual_test_ids,
        "all_split_views_resolved_to_full_colmap": all(item is not None for item in actual_train + actual_test),
        "train_exact_match": actual_train_global == expected_train_global,
        "test_exact_match": actual_test_global == expected_test_global,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a SparseGS-style 3DGS posed-view split.")
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--full_source", default="")
    parser.add_argument("--expected_train_views", type=int, default=0)
    parser.add_argument("--expected_hold", type=int, default=8)
    parser.add_argument("--require_sparsegs_triangulate", action="store_true")
    parser.add_argument("--no_train", action="store_true", help="Accepted for audit command compatibility; no training is run.")
    args = parser.parse_args()

    split_root = os.path.abspath(args.split_root)
    report_path = os.path.join(split_root, "reports", "split_report.json")
    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))

    check("split_report.json exists", os.path.isfile(report_path))
    if not os.path.isfile(report_path):
        for name, passed in checks:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print("[SPLIT-VALIDATE] status=FAIL")
        return 1

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    full_view_count = int(report.get("full_view_count", 0))
    expected_test = len(range(0, full_view_count, int(args.expected_hold)))
    train_source_path = report.get("train_source_path", os.path.join(split_root, "train"))
    test_source_path = report.get("test_source_path", os.path.join(split_root, "test"))
    train_sparse = os.path.join(train_source_path, "sparse", "0")
    test_sparse = os.path.join(test_source_path, "sparse", "0")
    sparsegs = report.get("sparsegs_triangulate", {})
    contamination = report.get("contamination_guard", {})

    check("status=PASS", report.get("status") == "PASS")
    check("split_unit=COLMAP_posed_view", report.get("split_unit") == "COLMAP_posed_view")
    check("test_view_count == len(range(0, full_view_count, hold))", int(report.get("test_view_count", -1)) == expected_test)
    if args.expected_train_views > 0:
        check("train_view_count == expected_train_views", int(report.get("train_view_count", -1)) == int(args.expected_train_views))
    check("overlap_by_image_name_count == 0", int(report.get("overlap_by_image_name_count", -1)) == 0)
    check("overlap_by_colmap_image_id_count == 0", int(report.get("overlap_by_colmap_image_id_count", -1)) == 0)
    check("duplicate_train_name_count == 0", int(report.get("duplicate_train_name_count", -1)) == 0)
    check("duplicate_test_name_count == 0", int(report.get("duplicate_test_name_count", -1)) == 0)
    check("local_cluster_guard_pass == true", report.get("sequence_coverage", {}).get("local_cluster_guard_pass") is True)
    check("train/images count == train_view_count", _count_images(os.path.join(train_source_path, "images")) == int(report.get("train_view_count", -1)))
    check("test/images count == test_view_count", _count_images(os.path.join(test_source_path, "images")) == int(report.get("test_view_count", -1)))
    check("train/sparse/0/cameras.bin exists", os.path.isfile(os.path.join(train_sparse, "cameras.bin")))
    check("train/sparse/0/images.bin exists", os.path.isfile(os.path.join(train_sparse, "images.bin")))
    check("train/sparse/0/points3D.bin exists", os.path.isfile(os.path.join(train_sparse, "points3D.bin")))
    check("train/sparse/0/points3D.ply exists", os.path.isfile(os.path.join(train_sparse, "points3D.ply")))
    check("test/sparse/0/cameras.bin exists", os.path.isfile(os.path.join(test_sparse, "cameras.bin")))
    check("test/sparse/0/images.bin exists", os.path.isfile(os.path.join(test_sparse, "images.bin")))
    check("train_contact_sheet.jpg exists", os.path.isfile(os.path.join(split_root, "reports", "train_contact_sheet.jpg")))
    check("test_contact_sheet.jpg exists", os.path.isfile(os.path.join(split_root, "reports", "test_contact_sheet.jpg")))
    check("camera_split_plot.png exists", os.path.isfile(os.path.join(split_root, "reports", "camera_split_plot.png")))
    check("selected_views.csv exists", os.path.isfile(os.path.join(split_root, "reports", "selected_views.csv")))

    if args.require_sparsegs_triangulate:
        check("sparsegs_triangulate.status == PASS", sparsegs.get("status") == "PASS")
        check("full_points3D_used_for_train_initialization == false", sparsegs.get("full_points3D_used_for_train_initialization") is False)
        check(
            "triangulated_point_count >= split_min_triangulated_points",
            int(sparsegs.get("triangulated_point_count", 0)) >= int(report.get("split_min_triangulated_points", 100)),
        )
        check("contamination_guard.test_rgb_used_for_training == false", contamination.get("test_rgb_used_for_training") is False)
        check("contamination_guard.internal_llffhold_disabled == true", contamination.get("internal_llffhold_disabled") is True)

    recomputed = None
    if args.full_source:
        recomputed = _recompute_exact_split(
            full_source=os.path.abspath(args.full_source),
            split_root=split_root,
            expected_hold=args.expected_hold,
            expected_train_views=args.expected_train_views or int(report.get("train_view_count", 0)),
        )
        check("full_source images.bin exact recompute resolved split views", recomputed["all_split_views_resolved_to_full_colmap"])
        check("actual test global indices == every expected_hold view", recomputed["test_exact_match"])
        check("actual train global indices == numpy_round linspace train pool", recomputed["train_exact_match"])

    library_validation = validate_split_report(report)
    check("library validation PASS", library_validation.get("status") == "PASS")

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"[SPLIT-VALIDATE] status={'PASS' if not failed else 'FAIL'}")
    if recomputed is not None:
        print("[SPLIT-VALIDATE] recomputed_exact_split=" + json.dumps(recomputed, separators=(",", ":")))
    if failed:
        print("[SPLIT-VALIDATE] failed_checks=" + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
