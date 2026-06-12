import argparse
import importlib
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    suffix = f" {detail}" if detail else ""
    print(f"[PREFLIGHT][{status}] {name}{suffix}")
    return bool(condition)


def _resolve_colmap(colmap_exe):
    if colmap_exe and colmap_exe.lower() != "colmap":
        return colmap_exe if os.path.isfile(colmap_exe) else shutil.which(colmap_exe)
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
        found = shutil.which(candidate)
        if found:
            return found
    return None


def main():
    parser = argparse.ArgumentParser(description="Preflight checks for strict auto split training.")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--split_train_views", default="12")
    parser.add_argument("--split_hold", type=int, default=8)
    parser.add_argument("--split_output_root", required=True)
    parser.add_argument("--split_name", default="")
    parser.add_argument("--split_copy_mode", default="copy")
    parser.add_argument("--split_force", action="store_true")
    parser.add_argument("--split_train_sample_mode", default="paper_even")
    parser.add_argument("--split_init_policy", default="sparsegs_triangulate")
    parser.add_argument("--split_colmap_exe", default="colmap")
    parser.add_argument("--split_colmap_matcher", default="exhaustive")
    parser.add_argument("--split_min_train_points", type=int, default=100)
    parser.add_argument("--split_min_triangulated_points", type=int, default=100)
    parser.add_argument("--metrics_compute_lpips", action="store_true")
    parser.add_argument("--run_split_validate", action="store_true")
    args = parser.parse_args()

    checks = []
    for module_name in (
        "utils.view_split_utils",
        "utils.colmap_split_io",
        "utils.colmap_sparsegs_triangulate",
        "utils.train_only_colmap_init",
        "utils.auto_split_3dgs",
    ):
        try:
            importlib.import_module(module_name)
            checks.append(_check(f"import {module_name}", True))
        except Exception as exc:
            checks.append(_check(f"import {module_name}", False, str(exc)))

    source_path = os.path.abspath(args.source_path)
    checks.append(_check("dataset full exists", os.path.isdir(source_path), source_path))
    checks.append(_check("full/images exists", os.path.isdir(os.path.join(source_path, "images"))))
    for rel in ("sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin"):
        checks.append(_check(f"full/{rel} exists", os.path.isfile(os.path.join(source_path, rel))))

    if args.split_init_policy in {"sparsegs_triangulate", "train_only_mapper", "train_only_colmap"}:
        colmap_path = _resolve_colmap(args.split_colmap_exe)
        checks.append(_check("colmap executable found", bool(colmap_path), colmap_path or ""))

    if args.metrics_compute_lpips:
        try:
            import lpips  # noqa: F401
            checks.append(_check("lpips import", True))
        except Exception as exc:
            checks.append(_check("lpips import", False, str(exc)))

    try:
        import diff_gaussian_rasterization  # noqa: F401
        checks.append(_check("diff_gaussian_rasterization import", True))
    except Exception as exc:
        checks.append(_check("diff_gaussian_rasterization import", False, str(exc)))

    if args.run_split_validate and all(checks):
        from utils.auto_split_3dgs import prepare_auto_split

        result = prepare_auto_split(
            source_path=source_path,
            split_train_views=args.split_train_views,
            split_hold=args.split_hold,
            split_output_root=args.split_output_root,
            split_name=args.split_name,
            split_copy_mode=args.split_copy_mode,
            split_force=args.split_force,
            split_train_sample_mode=args.split_train_sample_mode,
            split_init_policy=args.split_init_policy,
            split_colmap_exe=args.split_colmap_exe,
            split_colmap_matcher=args.split_colmap_matcher,
            split_require_all_train_registered=True,
            split_min_train_points=args.split_min_train_points,
            split_min_triangulated_points=args.split_min_triangulated_points,
        )
        checks.append(_check("split validate-only PASS", result.get("status") == "PASS", str(result)))

    passed = all(checks)
    print(f"[PREFLIGHT] status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
