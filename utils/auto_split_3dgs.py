import json
import os
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from utils.colmap_split_io import (
    copy_view_images,
    images_for_views,
    subset_cameras_for_images,
    write_pose_only_sparse_model,
    write_train_sparse_model,
)
from utils.colmap_sparsegs_triangulate import run_sparsegs_triangulate_init
from utils.train_only_colmap_init import run_train_only_colmap
from utils.view_split_utils import (
    compute_pose_audit,
    compute_sequence_coverage,
    compute_view_split,
    load_full_colmap_views,
    natural_lower_key,
    normalized_rel_name,
    selected_ids,
    selected_names,
    write_selected_views_csv,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROTOCOL = "sparsegs_mipnerf360_every8_test_even_train_views"


def json_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def _write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def _write_lines(path: str, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(str(value) + "\n")


def _safe_remove_tree(target: str, allowed_root: str):
    target_real = os.path.realpath(os.path.abspath(target))
    root_real = os.path.realpath(os.path.abspath(allowed_root))
    if target_real == root_real:
        raise RuntimeError(f"refusing to remove split output root itself: {target_real}")
    if os.path.commonpath([target_real, root_real]) != root_real:
        raise RuntimeError(f"refusing to remove path outside split output root: {target_real}")
    shutil.rmtree(target_real)


def _list_images(path: str):
    if not os.path.isdir(path):
        return []
    out = []
    for root, _, files in os.walk(path):
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                out.append(os.path.join(root, filename))
    return sorted(out, key=natural_lower_key)


def _duplicate_count(values: Sequence[str]) -> int:
    return len(values) - len(set(values))


def _empty_report(source_path: str, split_root: str, reports_dir: str, reason: str):
    return {
        "status": "FAIL",
        "protocol": PROTOCOL,
        "source_path": json_path(source_path),
        "split_root": json_path(split_root),
        "train_source_path": json_path(os.path.join(split_root, "train")),
        "test_source_path": json_path(os.path.join(split_root, "test")),
        "split_report_path": json_path(os.path.join(reports_dir, "split_report.json")),
        "validation_report_path": json_path(os.path.join(reports_dir, "validation_report.json")),
        "reason": reason,
    }


def _write_split_report_text(path: str, report: dict, validation_report: dict):
    lines = [
        f"status: {report.get('status')}",
        f"protocol: {report.get('protocol')}",
        f"source_path: {report.get('source_path')}",
        f"split_root: {report.get('split_root')}",
        f"train_source_path: {report.get('train_source_path')}",
        f"test_source_path: {report.get('test_source_path')}",
        f"split_hold: {report.get('split_hold')}",
        f"split_train_views: {report.get('split_train_views')}",
        f"split_train_sample_mode: {report.get('split_train_sample_mode')}",
        f"split_init_policy: {report.get('split_init_policy')}",
        f"full_view_count: {report.get('full_view_count')}",
        f"test_view_count: {report.get('test_view_count')}",
        f"train_pool_view_count: {report.get('train_pool_view_count')}",
        f"train_view_count: {report.get('train_view_count')}",
        f"overlap_by_image_name_count: {report.get('overlap_by_image_name_count')}",
        f"overlap_by_colmap_image_id_count: {report.get('overlap_by_colmap_image_id_count')}",
        f"local_cluster_guard_pass: {report.get('sequence_coverage', {}).get('local_cluster_guard_pass')}",
        f"pose_coverage_guard_pass: {report.get('pose_audit', {}).get('pose_coverage_guard_pass')}",
        f"sparsegs_triangulate.status: {report.get('sparsegs_triangulate', {}).get('status')}",
        f"triangulated_point_count: {report.get('sparsegs_triangulate', {}).get('triangulated_point_count')}",
        f"train_only_mapper.status: {report.get('train_only_mapper', {}).get('status')}",
        f"aligned_point_count: {report.get('train_only_mapper', {}).get('aligned_point_count')}",
        f"validation_status: {validation_report.get('status')}",
        f"split_report_path: {report.get('split_report_path')}",
        f"validation_report_path: {report.get('validation_report_path')}",
    ]
    failed = validation_report.get("failed_checks", [])
    if failed:
        lines.append("failed_checks: " + ", ".join(failed))
    _write_lines(path, lines)


def _make_contact_sheet(views, output_path: str, thumb_size=(180, 120), max_cols=6):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        _write_lines(output_path + ".txt", ["PIL unavailable; contact sheet not generated"])
        return False

    if not views:
        canvas = Image.new("RGB", (thumb_size[0], thumb_size[1]), "white")
        canvas.save(output_path)
        return True

    cols = min(max_cols, max(1, len(views)))
    rows = int(np.ceil(len(views) / cols))
    label_h = 34
    canvas = Image.new("RGB", (cols * thumb_size[0], rows * (thumb_size[1] + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, view in enumerate(views):
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * (thumb_size[1] + label_h)
        try:
            with Image.open(view.image_path) as img:
                img = img.convert("RGB")
                img.thumbnail(thumb_size)
                px = x + (thumb_size[0] - img.size[0]) // 2
                py = y + (thumb_size[1] - img.size[1]) // 2
                canvas.paste(img, (px, py))
        except Exception:
            draw.rectangle([x, y, x + thumb_size[0] - 1, y + thumb_size[1] - 1], outline=(180, 0, 0))
        label = f"{view.global_index}: {Path(view.image_name).name}"
        draw.text((x + 4, y + thumb_size[1] + 4), label[:32], fill=(0, 0, 0), font=font)

    canvas.save(output_path)
    return True


def _make_camera_plot(sorted_views, train_views, test_views, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        from PIL import Image, ImageDraw
    except Exception:
        _write_lines(output_path + ".txt", ["PIL unavailable; camera plot not generated"])
        return False

    width, height = 1000, 720
    margin = 60
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    centers = np.asarray([v.camera_center for v in sorted_views], dtype=np.float64)
    if centers.size == 0:
        canvas.save(output_path)
        return True

    xy = centers[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-9)

    def project(view):
        point = np.asarray(view.camera_center[:2], dtype=np.float64)
        norm = (point - min_xy) / span
        x = margin + norm[0] * (width - 2 * margin)
        y = height - margin - norm[1] * (height - 2 * margin)
        return float(x), float(y)

    train_ids = {v.colmap_image_id for v in train_views}
    test_ids = {v.colmap_image_id for v in test_views}

    for view in sorted_views:
        x, y = project(view)
        color = (170, 170, 170)
        radius = 2
        if view.colmap_image_id in test_ids:
            color = (230, 130, 40)
            radius = 3
        if view.colmap_image_id in train_ids:
            color = (35, 90, 210)
            radius = 6
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=color)

    draw.text((margin, 20), "gray=full, blue=train, orange=test", fill=(0, 0, 0))
    canvas.save(output_path)
    return True


def _subset_full_points_for_views(points3D: dict, selected_views):
    selected_image_ids = {int(v.colmap_image_id) for v in selected_views}
    out = {}
    for point_id, point in points3D.items():
        image_ids = np.asarray(point.image_ids, dtype=np.int32)
        if any(int(image_id) in selected_image_ids for image_id in image_ids):
            out[int(point_id)] = point
    return out


def _base_assertions(report: dict):
    train_only_mapper = report.get("train_only_mapper", {})
    sparsegs_triangulate = report.get("sparsegs_triangulate", {})
    assertions = {
        "full_views_match_images": (
            report.get("missing_image_file_count") == 0
            and report.get("full_view_count") == report.get("full_colmap_image_count")
            and report.get("full_colmap_image_count") == report.get("full_image_file_count")
        ),
        "test_count_matches_hold": report.get("test_view_count") == report.get("expected_test_view_count"),
        "train_count_matches_request": (
            report.get("train_view_count") == report.get("train_pool_view_count")
            if str(report.get("split_train_views")) == "full"
            else report.get("train_view_count") == int(report.get("split_train_views"))
        ),
        "overlap_zero_by_name": report.get("overlap_by_image_name_count") == 0,
        "overlap_zero_by_colmap_id": report.get("overlap_by_colmap_image_id_count") == 0,
        "duplicate_train_zero": report.get("duplicate_train_name_count") == 0,
        "duplicate_test_zero": report.get("duplicate_test_name_count") == 0,
        "local_cluster_guard_pass": bool(report.get("sequence_coverage", {}).get("local_cluster_guard_pass")),
        "pose_coverage_guard_pass": bool(report.get("pose_audit", {}).get("pose_coverage_guard_pass")),
        "external_test_reader_required": bool(report.get("contamination_guard", {}).get("external_test_source_enabled")),
        "no_legacy_llff_resplit": bool(report.get("contamination_guard", {}).get("internal_llffhold_disabled")),
    }

    if report.get("split_init_policy") == "sparsegs_triangulate":
        assertions.update(
            {
                "sparsegs_triangulate_pass": sparsegs_triangulate.get("status") == "PASS",
                "triangulated_points_enough": int(sparsegs_triangulate.get("triangulated_point_count", 0))
                >= int(sparsegs_triangulate.get("min_required_points", report.get("split_min_triangulated_points", 0))),
                "full_points3D_not_used_for_train_initialization": sparsegs_triangulate.get("full_points3D_used_for_train_initialization") is False,
            }
        )
    if report.get("split_init_policy") == "train_only_mapper":
        assertions.update(
            {
                "train_only_mapper_pass": train_only_mapper.get("status") == "PASS",
                "all_train_views_registered": train_only_mapper.get("all_train_views_registered") is True,
                "aligned_train_points_enough": int(train_only_mapper.get("aligned_point_count", 0)) >= int(report.get("split_min_train_points", 0)),
            }
        )
    return assertions


def validate_split_report(report: dict, strict_no_overlap=True):
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    source_path = report.get("source_path", "")
    train_source_path = report.get("train_source_path", "")
    test_source_path = report.get("test_source_path", "")
    train_sparse = os.path.join(train_source_path, "sparse", "0")
    test_sparse = os.path.join(test_source_path, "sparse", "0")

    check("status_pass", report.get("status") == "PASS")
    check("protocol_correct", report.get("protocol") == PROTOCOL)
    check("source_full_exists", os.path.isdir(source_path))
    check("full_images_exists", os.path.isdir(os.path.join(source_path, "images")))
    check("full_sparse_cameras_bin_exists", os.path.isfile(os.path.join(source_path, "sparse", "0", "cameras.bin")))
    check("full_sparse_images_bin_exists", os.path.isfile(os.path.join(source_path, "sparse", "0", "images.bin")))
    check("full_sparse_points3D_bin_exists", os.path.isfile(os.path.join(source_path, "sparse", "0", "points3D.bin")))
    check("full_view_count_positive", int(report.get("full_view_count", 0)) > 0)
    check("full_views_match_images", report.get("assertions", {}).get("full_views_match_images") is True)
    check("test_count_matches_hold", report.get("assertions", {}).get("test_count_matches_hold") is True)
    check("train_count_matches_request", report.get("assertions", {}).get("train_count_matches_request") is True)
    check("overlap_zero_by_name", report.get("assertions", {}).get("overlap_zero_by_name") is True if strict_no_overlap else True)
    check("overlap_zero_by_colmap_id", report.get("assertions", {}).get("overlap_zero_by_colmap_id") is True if strict_no_overlap else True)
    check("duplicate_train_zero", report.get("assertions", {}).get("duplicate_train_zero") is True)
    check("duplicate_test_zero", report.get("assertions", {}).get("duplicate_test_zero") is True)
    check("local_cluster_guard_pass", report.get("assertions", {}).get("local_cluster_guard_pass") is True)
    check("pose_coverage_guard_pass", report.get("assertions", {}).get("pose_coverage_guard_pass") is True)
    check("train_folder_image_count", len(_list_images(os.path.join(train_source_path, "images"))) == int(report.get("train_view_count", -1)))
    check("test_folder_image_count", len(_list_images(os.path.join(test_source_path, "images"))) == int(report.get("test_view_count", -1)))
    check("train_sparse_cameras_bin_exists", os.path.isfile(os.path.join(train_sparse, "cameras.bin")))
    check("train_sparse_images_bin_exists", os.path.isfile(os.path.join(train_sparse, "images.bin")))
    check("train_sparse_points3D_bin_exists", os.path.isfile(os.path.join(train_sparse, "points3D.bin")))
    check("train_sparse_points3D_ply_exists", os.path.isfile(os.path.join(train_sparse, "points3D.ply")))
    check("test_sparse_cameras_bin_exists", os.path.isfile(os.path.join(test_sparse, "cameras.bin")))
    check("test_sparse_images_bin_exists", os.path.isfile(os.path.join(test_sparse, "images.bin")))
    check("test_sparse_points3D_bin_exists", os.path.isfile(os.path.join(test_sparse, "points3D.bin")))
    check("selected_views_csv_exists", os.path.isfile(os.path.join(report.get("split_root", ""), "reports", "selected_views.csv")))
    check("train_contact_sheet_exists", os.path.isfile(os.path.join(report.get("split_root", ""), "reports", "train_contact_sheet.jpg")))
    check("test_contact_sheet_exists", os.path.isfile(os.path.join(report.get("split_root", ""), "reports", "test_contact_sheet.jpg")))
    check("camera_split_plot_exists", os.path.isfile(os.path.join(report.get("split_root", ""), "reports", "camera_split_plot.png")))
    check("external_test_source_enabled", report.get("contamination_guard", {}).get("external_test_source_enabled") is True)
    check("internal_llffhold_disabled", report.get("contamination_guard", {}).get("internal_llffhold_disabled") is True)
    check("nerf_normalization_train_only", report.get("contamination_guard", {}).get("nerf_normalization_source") == "train_cameras_only")

    if report.get("split_init_policy") == "sparsegs_triangulate":
        sparsegs_triangulate = report.get("sparsegs_triangulate", {})
        check("sparsegs_triangulate_pass", sparsegs_triangulate.get("status") == "PASS")
        check(
            "triangulated_points_enough",
            int(sparsegs_triangulate.get("triangulated_point_count", 0))
            >= int(sparsegs_triangulate.get("min_required_points", report.get("split_min_triangulated_points", 0))),
        )
        check("full_points3D_not_used_for_train_initialization", sparsegs_triangulate.get("full_points3D_used_for_train_initialization") is False)
        check(
            "point_cloud_source_sparsegs_triangulate",
            report.get("contamination_guard", {}).get("initial_point_cloud_source")
            == "sparsegs_triangulate_train_views_only",
        )

    if report.get("split_init_policy") == "train_only_mapper":
        train_only_mapper = report.get("train_only_mapper", {})
        check("train_only_mapper_pass", train_only_mapper.get("status") == "PASS")
        check("all_train_views_registered", train_only_mapper.get("all_train_views_registered") is True)
        check(
            "aligned_train_points_enough",
            int(train_only_mapper.get("aligned_point_count", 0)) >= int(report.get("split_min_train_points", 0)),
        )
        check(
            "point_cloud_source_train_only",
            report.get("contamination_guard", {}).get("initial_point_cloud_source")
            == "train_only_mapper_selected_train_views",
        )

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
    }


def validate_split_report_file(report_path: str, strict_no_overlap=True):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return validate_split_report(report, strict_no_overlap=strict_no_overlap)


def _write_fail(source_path, split_root, reports_dir, reason, extra=None):
    os.makedirs(reports_dir, exist_ok=True)
    report = _empty_report(source_path, split_root, reports_dir, reason)
    if extra:
        report.update(extra)
    report_path = os.path.join(reports_dir, "split_report.json")
    validation_path = os.path.join(reports_dir, "validation_report.json")
    validation = {
        "status": "FAIL",
        "checks": {"exception_free": False},
        "failed_checks": ["exception_free"],
        "reason": reason,
    }
    _write_json(report_path, report)
    _write_json(validation_path, validation)
    _write_split_report_text(os.path.join(reports_dir, "split_report.txt"), report, validation)
    return {
        "status": "FAIL",
        "reason": reason,
        "split_root": report["split_root"],
        "train_source_path": report["train_source_path"],
        "test_source_path": report["test_source_path"],
        "split_report_path": report["split_report_path"],
        "validation_report_path": report["validation_report_path"],
    }


def prepare_auto_split(
    source_path: str,
    split_train_views: str,
    split_hold: int,
    split_output_root: str = "",
    split_name: str = "",
    split_copy_mode: str = "copy",
    split_force: bool = False,
    split_train_sample_mode: str = "paper_even",
    split_init_policy: str = "sparsegs_triangulate",
    split_colmap_exe: str = "colmap",
    split_colmap_matcher: str = "exhaustive",
    split_require_all_train_registered: bool = True,
    split_min_train_points: int = None,
    split_min_triangulated_points: int = 100,
    split_strict_sparsegs: bool = True,
    strict_no_overlap: bool = True,
    # Backward-compatible names used by the previous local implementation.
    output_root: str = None,
    copy_mode: str = None,
    force: bool = None,
    init_policy: str = None,
) -> dict:
    if output_root is not None:
        split_output_root = output_root
    if copy_mode is not None:
        split_copy_mode = copy_mode
    if force is not None:
        split_force = force
    if init_policy is not None:
        split_init_policy = init_policy

    valid_train_views = {"full", "3", "6", "9", "12", "24"}
    valid_copy_modes = {"copy", "hardlink", "symlink"}
    valid_sample_modes = {"paper_even", "pose_fps"}
    valid_init_policies = {"sparsegs_triangulate", "subset_colmap", "train_only_mapper", "train_only_colmap"}

    source_path = os.path.abspath(source_path)
    split_train_views = str(split_train_views)
    split_hold = int(split_hold)
    if split_min_train_points is None:
        split_min_train_points = split_min_triangulated_points
    split_min_train_points = int(split_min_train_points)
    split_min_triangulated_points = int(split_min_triangulated_points)

    if split_init_policy == "train_only_colmap":
        print(
            "[DEPRECATED] --split_init_policy train_only_colmap is ambiguous. "
            "Mapping it to train_only_mapper. Use sparsegs_triangulate for SparseGS-style protocol."
        )
        split_init_policy = "train_only_mapper"

    if split_train_views not in valid_train_views:
        raise ValueError(f"split_train_views must be one of {sorted(valid_train_views)}, got {split_train_views!r}")
    if split_copy_mode not in valid_copy_modes:
        raise ValueError(f"split_copy_mode must be one of {sorted(valid_copy_modes)}, got {split_copy_mode!r}")
    if split_train_sample_mode not in valid_sample_modes:
        raise ValueError(f"split_train_sample_mode must be one of {sorted(valid_sample_modes)}, got {split_train_sample_mode!r}")
    if split_init_policy not in valid_init_policies:
        raise ValueError(f"split_init_policy must be one of {sorted(valid_init_policies)}, got {split_init_policy!r}")

    if not split_output_root:
        split_output_root = os.path.join(source_path, "_3dgs_splits")
    split_output_root = os.path.abspath(split_output_root)
    folder_name = split_name if split_name else f"hold{split_hold}_train{split_train_views}_{split_init_policy}"
    split_root = os.path.abspath(os.path.join(split_output_root, folder_name))
    reports_dir = os.path.join(split_root, "reports")
    train_source_path = os.path.join(split_root, "train")
    test_source_path = os.path.join(split_root, "test")
    split_report_path = os.path.join(reports_dir, "split_report.json")
    validation_report_path = os.path.join(reports_dir, "validation_report.json")
    split_report_txt_path = os.path.join(reports_dir, "split_report.txt")

    try:
        if os.path.exists(split_root):
            if not split_force:
                return _write_fail(
                    source_path,
                    split_root,
                    reports_dir,
                    "split folder already exists; use --split_force to overwrite",
                )
            _safe_remove_tree(split_root, split_output_root)
        os.makedirs(reports_dir, exist_ok=True)

        sorted_views, full_cameras, full_images, full_points3D, source_audit = load_full_colmap_views(
            source_path,
            images_dir_name="images",
            return_model=True,
        )
        if source_audit["missing_image_file_count"] or source_audit["ambiguous_image_name_count"]:
            return _write_fail(
                source_path,
                split_root,
                reports_dir,
                "COLMAP posed views do not map cleanly to source RGB files",
                extra=source_audit,
            )

        selected_train, test_views, train_pool, train_pool_positions = compute_view_split(
            sorted_views,
            hold=split_hold,
            train_views=split_train_views,
            sample_mode=split_train_sample_mode,
        )

        sequence_coverage = compute_sequence_coverage(selected_train, sorted_views)
        pose_audit = compute_pose_audit(selected_train, sorted_views)
        if not sequence_coverage["local_cluster_guard_pass"]:
            return _write_fail(
                source_path,
                split_root,
                reports_dir,
                "selected train views are a local cluster, not a sparse trajectory sample",
                extra={"sequence_coverage": sequence_coverage},
            )
        if not pose_audit["pose_coverage_guard_pass"]:
            return _write_fail(
                source_path,
                split_root,
                reports_dir,
                "selected train camera centers have insufficient pose coverage",
                extra={"pose_audit": pose_audit},
            )

        os.makedirs(train_source_path, exist_ok=True)
        os.makedirs(test_source_path, exist_ok=True)
        copy_view_images(selected_train, train_source_path, split_copy_mode)
        copy_view_images(test_views, test_source_path, split_copy_mode)

        train_images_subset = images_for_views(full_images, selected_train)
        test_images_subset = images_for_views(full_images, test_views)
        train_cameras_subset = subset_cameras_for_images(full_cameras, train_images_subset.values())
        test_cameras_subset = subset_cameras_for_images(full_cameras, test_images_subset.values())

        test_model_stats = write_pose_only_sparse_model(
            os.path.join(test_source_path, "sparse", "0"),
            test_cameras_subset,
            test_images_subset,
        )

        sparsegs_triangulate_report = {
            "status": "SKIPPED",
            "reason": f"split_init_policy is {split_init_policy}",
        }
        train_only_mapper_report = {
            "status": "SKIPPED",
            "reason": f"split_init_policy is {split_init_policy}",
        }

        if split_init_policy == "sparsegs_triangulate":
            sparsegs_triangulate_report = run_sparsegs_triangulate_init(
                full_source_path=source_path,
                split_root=split_root,
                train_views=selected_train,
                test_views=test_views,
                colmap_exe=split_colmap_exe,
                matcher=split_colmap_matcher,
                min_points=split_min_triangulated_points,
                force=split_force,
            )
            train_model_stats = {
                "camera_count": int(sparsegs_triangulate_report.get("triangulated_camera_count", 0)),
                "image_count": int(sparsegs_triangulate_report.get("triangulated_image_count", 0)),
                "point_count": int(sparsegs_triangulate_report.get("triangulated_point_count", 0)),
            }
        elif split_init_policy == "train_only_mapper":
            train_only_result = run_train_only_colmap(
                train_images_dir=os.path.join(train_source_path, "images"),
                work_dir=os.path.join(split_root, "work", "train_only_mapper"),
                colmap_exe=split_colmap_exe,
                matcher=split_colmap_matcher,
                selected_train_views=selected_train,
                full_reference_views_by_name={view.image_name: view for view in sorted_views},
                require_all_registered=split_require_all_train_registered,
                min_aligned_points=split_min_train_points,
            )
            train_only_mapper_report = dict(train_only_result["report"])
            train_only_mapper_report["policy"] = "train_only_mapper"
            train_points = train_only_result["aligned_points3D"]
            train_model_stats = write_train_sparse_model(
                os.path.join(train_source_path, "sparse", "0"),
                train_cameras_subset,
                train_images_subset,
                train_points,
            )
        elif split_init_policy == "subset_colmap":
            train_points = _subset_full_points_for_views(full_points3D, selected_train)
            train_only_mapper_report = {
                "status": "SKIPPED",
                "raw_point_count": len(train_points),
                "aligned_point_count": len(train_points),
                "requested_train_view_count": len(selected_train),
                "registered_train_view_count": len(selected_train),
                "all_train_views_registered": True,
            }
            train_model_stats = write_train_sparse_model(
                os.path.join(train_source_path, "sparse", "0"),
                train_cameras_subset,
                train_images_subset,
                train_points,
            )
        else:
            raise ValueError(f"unknown split_init_policy after normalization: {split_init_policy}")

        train_names_norm = [normalized_rel_name(v.image_name) for v in selected_train]
        test_names_norm = [normalized_rel_name(v.image_name) for v in test_views]
        train_ids = selected_ids(selected_train)
        test_ids = selected_ids(test_views)
        overlap_names = sorted(set(train_names_norm) & set(test_names_norm), key=natural_lower_key)
        overlap_ids = sorted(set(train_ids) & set(test_ids))

        contamination_guard = {
            "test_rgb_used_for_training": False,
            "test_points_used_for_initialization": False,
            "full_points3D_used_for_train_initialization": split_init_policy == "subset_colmap",
            "initial_point_cloud_source": (
                "sparsegs_triangulate_train_views_only"
                if split_init_policy == "sparsegs_triangulate"
                else (
                    "train_only_mapper_selected_train_views"
                    if split_init_policy == "train_only_mapper"
                    else "subset_full_colmap_points_from_selected_train_views"
                )
            ),
            "pose_source": "full_colmap_reference_for_train_and_test_camera_poses",
            "nerf_normalization_source": "train_cameras_only",
            "internal_llffhold_disabled": True,
            "external_test_source_enabled": True,
            "strict_rgb_and_points_from_train_views": split_init_policy in {"sparsegs_triangulate", "train_only_mapper"},
            "all_camera_poses_assumed_known": True,
        }

        report = {
            "status": "PENDING",
            "protocol": PROTOCOL,
            "split_unit": "COLMAP_posed_view",
            "mode": "auto_full_dataset_holdout_split",
            "source_path": json_path(source_path),
            "split_root": json_path(split_root),
            "train_source_path": json_path(train_source_path),
            "test_source_path": json_path(test_source_path),
            "split_report_path": json_path(split_report_path),
            "validation_report_path": json_path(validation_report_path),
            "sort_mode": "natural_lowercase_colmap_image_name",
            "split_hold": split_hold,
            "effective_hold": split_hold,
            "test_rule": f"every_{split_hold}th_sorted_posed_view",
            "train_pool_rule": "all_non_test_posed_views",
            "train_sample_rule": "evenly_sample_K_from_train_pool" if split_train_sample_mode == "paper_even" else "pose_fps_sample_K_from_train_pool",
            "split_train_views": split_train_views,
            "split_train_sample_mode": split_train_sample_mode,
            "split_init_policy": split_init_policy,
            "init_policy": split_init_policy,
            "split_copy_mode": split_copy_mode,
            "split_colmap_exe": split_colmap_exe,
            "split_colmap_matcher": split_colmap_matcher,
            "split_require_all_train_registered": bool(split_require_all_train_registered),
            "split_min_train_points": split_min_train_points,
            "split_min_triangulated_points": split_min_triangulated_points,
            "split_strict_sparsegs": bool(split_strict_sparsegs),
            "full_view_count": len(sorted_views),
            "full_image_file_count": source_audit["full_image_file_count"],
            "full_colmap_image_count": source_audit["full_colmap_image_count"],
            "full_camera_count": len(full_cameras),
            "missing_image_file_count": source_audit["missing_image_file_count"],
            "ambiguous_image_name_count": source_audit["ambiguous_image_name_count"],
            "expected_test_view_count": len(range(0, len(sorted_views), split_hold)),
            "expected_test_count": len(range(0, len(sorted_views), split_hold)),
            "test_view_count": len(test_views),
            "test_count": len(test_views),
            "train_pool_view_count": len(train_pool),
            "train_pool_count": len(train_pool),
            "train_view_count": len(selected_train),
            "train_count": len(selected_train),
            "train_pool_positions": [int(p) for p in train_pool_positions],
            "train_global_indices": [int(v.global_index) for v in selected_train],
            "test_global_indices": [int(v.global_index) for v in test_views],
            "train_indices": [int(v.global_index) for v in selected_train],
            "test_indices": [int(v.global_index) for v in test_views],
            "train_colmap_image_ids": train_ids,
            "test_colmap_image_ids": test_ids,
            "train_image_names": selected_names(selected_train),
            "test_image_names": selected_names(test_views),
            "overlap_by_image_name_count": len(overlap_names),
            "overlap_by_image_names": overlap_names,
            "overlap_by_colmap_image_id_count": len(overlap_ids),
            "overlap_by_colmap_image_ids": overlap_ids,
            "overlap_count": len(overlap_names),
            "duplicate_train_name_count": _duplicate_count(train_names_norm),
            "duplicate_test_name_count": _duplicate_count(test_names_norm),
            "duplicate_train_count": _duplicate_count(train_names_norm),
            "duplicate_test_count": _duplicate_count(test_names_norm),
            "sequence_coverage": sequence_coverage,
            "pose_audit": pose_audit,
            "sparsegs_triangulate": sparsegs_triangulate_report,
            "train_only_mapper": train_only_mapper_report,
            "colmap_train_only": train_only_mapper_report,
            "contamination_guard": contamination_guard,
            "point_cloud_source": contamination_guard["initial_point_cloud_source"],
            "nerf_normalization_source": "train_cameras_only",
            "internal_llffhold_disabled": True,
            "external_test_source_enabled": True,
            "full_scene_point_cloud_used_for_training": False,
            "full_points3D_used_for_train_initialization": contamination_guard["full_points3D_used_for_train_initialization"],
            "train_model_stats": train_model_stats,
            "test_model_stats": test_model_stats,
            "pose_protocol": "posed_sparse_view_nvs",
            "pose_source": "full_colmap_reference_for_train_and_test_camera_poses",
            "initial_point_cloud_source": contamination_guard["initial_point_cloud_source"],
            "test_rgb_used_for_training": False,
            "test_points_used_for_initialization": False,
            "strict_rgb_and_points_from_train_views": contamination_guard["strict_rgb_and_points_from_train_views"],
            "all_camera_poses_assumed_known": True,
        }

        report["assertions"] = _base_assertions(report)
        failed_assertions = [key for key, value in report["assertions"].items() if not value]
        report["status"] = "PASS" if not failed_assertions else "FAIL"
        if failed_assertions:
            report["failure_reasons"] = failed_assertions

        _write_lines(os.path.join(reports_dir, "train_images.txt"), report["train_image_names"])
        _write_lines(os.path.join(reports_dir, "test_images.txt"), report["test_image_names"])
        _write_lines(os.path.join(reports_dir, "train_indices.txt"), report["train_global_indices"])
        _write_lines(os.path.join(reports_dir, "test_indices.txt"), report["test_global_indices"])
        write_selected_views_csv(os.path.join(reports_dir, "selected_views.csv"), selected_train, test_views)
        _make_contact_sheet(selected_train, os.path.join(reports_dir, "train_contact_sheet.jpg"))
        _make_contact_sheet(test_views, os.path.join(reports_dir, "test_contact_sheet.jpg"))
        _make_camera_plot(sorted_views, selected_train, test_views, os.path.join(reports_dir, "camera_split_plot.png"))
        _write_json(os.path.join(reports_dir, "sparsegs_triangulate_report.json"), sparsegs_triangulate_report)
        _write_json(os.path.join(reports_dir, "train_only_mapper_report.json"), train_only_mapper_report)

        validation_report = validate_split_report(report, strict_no_overlap=strict_no_overlap)
        if validation_report["status"] != "PASS":
            report["status"] = "FAIL"
            report["failure_reasons"] = validation_report["failed_checks"]
            validation_report = validate_split_report(report, strict_no_overlap=strict_no_overlap)

        _write_json(split_report_path, report)
        _write_json(validation_report_path, validation_report)
        _write_split_report_text(split_report_txt_path, report, validation_report)

        print(f"[AUTO-SPLIT] status={report['status']}")
        print(f"[AUTO-SPLIT] protocol={report['protocol']}")
        print(f"[AUTO-SPLIT] full_view_count={report['full_view_count']}")
        print(f"[AUTO-SPLIT] test_view_count={report['test_view_count']}")
        print(f"[AUTO-SPLIT] train_pool_view_count={report['train_pool_view_count']}")
        print(f"[AUTO-SPLIT] train_view_count={report['train_view_count']}")
        print(f"[AUTO-SPLIT] split_train_sample_mode={report['split_train_sample_mode']}")
        print(f"[AUTO-SPLIT] split_init_policy={report['split_init_policy']}")
        print(f"[AUTO-SPLIT] overlap_by_image_name_count={report['overlap_by_image_name_count']}")
        print(f"[AUTO-SPLIT] overlap_by_colmap_image_id_count={report['overlap_by_colmap_image_id_count']}")
        print(f"[AUTO-SPLIT] local_cluster_guard_pass={report['sequence_coverage']['local_cluster_guard_pass']}")
        print(f"[AUTO-SPLIT] sparsegs_triangulate.status={report['sparsegs_triangulate'].get('status')}")
        print(f"[AUTO-SPLIT] full_points3D_used_for_train_initialization={report['contamination_guard']['full_points3D_used_for_train_initialization']}")
        print(f"[AUTO-SPLIT] internal_llffhold_disabled={report['contamination_guard']['internal_llffhold_disabled']}")
        print(f"[AUTO-SPLIT] split_report={report['split_report_path']}")

        return {
            "status": report["status"],
            "reason": report.get("reason", ""),
            "failure_reasons": report.get("failure_reasons", []),
            "split_root": report["split_root"],
            "train_source_path": report["train_source_path"],
            "test_source_path": report["test_source_path"],
            "split_report_path": report["split_report_path"],
            "validation_report_path": report["validation_report_path"],
            "full_view_count": report["full_view_count"],
            "test_count": report["test_view_count"],
            "train_pool_count": report["train_pool_view_count"],
            "train_count": report["train_view_count"],
            "overlap_count": report["overlap_by_image_name_count"],
            "protocol": report["protocol"],
        }
    except Exception as exc:
        return _write_fail(source_path, split_root, reports_dir, str(exc))
