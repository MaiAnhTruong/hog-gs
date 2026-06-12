"""Build the aligned inverse-depth cache (schema PGDR_DEPTH_CACHE_V2_ALIGNED) for a split.

Input : Depth-Anything-V2 raw relative-inverse-depth .npy per train image (from
        tools/run_da2_depth.py) + the split's train COLMAP model (train/sparse/0).
Output: <split>/train/pgdr_depth_cache_aligned/
          depth_inv_aligned.pt          [V,1,H,W] float32, COLMAP-scale inverse depth
          depth_meta.json               frames[] matched by stem (consumed by utils.depthguide)
          pgdr_depth_build_report.json  build provenance
          sparse_alignment_report.json  per-view robust-fit diagnostics

Alignment: DA2 outputs RELATIVE inverse depth; COLMAP gives metric-scale sparse depth at the
view's triangulated observations. Per view we robust-fit  inv_colmap ~= a * raw + b  (least
squares -> MAD inlier gate at 3*1.4826*MAD -> refit on inliers, 2 rounds), falling back to a
global pooled fit when a view has too few observations or a degenerate (a<=0) fit. The raw map
is bilinearly resized to the training resolution (the standard 3DGS width-1600 rule) and the
affine fit is applied.

Usage:
  python tools/build_depth_cache.py --split_root <scene>/<split>   [--raw_dir ...] [--out_dir ...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.read_write_model import read_model, qvec2rotmat

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SCHEMA = "PGDR_DEPTH_CACHE_V2_ALIGNED"


def _natural_lower_key(name):
    import re
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(name).lower())]


def _find_raw_dir(split_root, raw_dir_arg):
    if raw_dir_arg:
        return raw_dir_arg if os.path.isdir(raw_dir_arg) else None
    candidates = [os.path.join(split_root, "train", "depth_anything_v2_vitl", "raw_npy_float32")]
    candidates += glob.glob(os.path.join(split_root, "train", "*depth_anything*", "raw_npy_float32"))
    # legacy layout: raw folder next to the split, named <scene>_<split>_train_depth_anything_v2_*
    candidates += glob.glob(os.path.join(os.path.dirname(split_root), "*depth_anything*", "raw_npy_float32"))
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _train_resolution(images_dir, image_name):
    """Resolution train.py will use with -r -1: widths above 1600 are scaled down to 1600."""
    from PIL import Image
    with Image.open(os.path.join(images_dir, image_name)) as im:
        w, h = im.size
    if w > 1600:
        scale = w / 1600.0
        return int(w / scale), int(h / scale)
    return w, h


def _bilinear_sample(arr, x, y):
    """Sample 2D float array at float pixel coords (x right, y down). Returns nan out of bounds."""
    h, w = arr.shape
    if not (0.0 <= x <= w - 1 and 0.0 <= y <= h - 1):
        return float("nan")
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    v = (arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x1] * fx * (1 - fy)
         + arr[y1, x0] * (1 - fx) * fy + arr[y1, x1] * fx * fy)
    return float(v)


def _theil_sen(raw, inv, max_pairs=5000, rng_seed=0):
    """Median of pairwise slopes + median intercept. Breakdown ~29% and robust to LEVERAGE
    outliers (gross errors in the sampled raw depth, e.g. occlusion boundaries) where a least-
    squares init collapses the slope toward 0."""
    n = raw.size
    ii, jj = np.triu_indices(n, k=1)
    if ii.size > max_pairs:
        sel = np.random.default_rng(rng_seed).choice(ii.size, size=max_pairs, replace=False)
        ii, jj = ii[sel], jj[sel]
    dr = raw[jj] - raw[ii]
    keep = np.abs(dr) > 1e-12
    if not keep.any():
        return None
    a = float(np.median((inv[jj] - inv[ii])[keep] / dr[keep]))
    b = float(np.median(inv - a * raw))
    return a, b


def _robust_fit(raw, inv, rounds=2):
    """Robust affine fit inv ~= a*raw + b: Theil-Sen init -> MAD inlier gate -> least-squares
    refit on inliers (iterated). Returns (a, b, inlier_mask, stats) or None."""
    raw = np.asarray(raw, dtype=np.float64)
    inv = np.asarray(inv, dtype=np.float64)
    n = raw.size
    if n < 2:
        return None
    init = _theil_sen(raw, inv)
    if init is None:
        return None
    a, b = init
    inliers = np.ones(n, dtype=bool)
    for _ in range(rounds + 1):
        r = a * raw + b - inv
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        thr = max(3.0 * 1.4826 * mad, 1e-9)
        inliers = np.abs(r - med) <= thr
        if inliers.sum() < 2:
            return None
        A = np.stack([raw[inliers], np.ones(int(inliers.sum()))], axis=1)
        sol, *_ = np.linalg.lstsq(A, inv[inliers], rcond=None)
        a, b = float(sol[0]), float(sol[1])
    res_in = np.abs(a * raw[inliers] + b - inv[inliers])
    corr = 0.0
    if inliers.sum() >= 2 and np.std(raw[inliers]) > 1e-12 and np.std(inv[inliers]) > 1e-12:
        corr = float(np.corrcoef(raw[inliers], inv[inliers])[0, 1])
    stats = {
        "num_inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "median_abs_residual_inv": float(np.median(res_in)) if res_in.size else 0.0,
        "mad_abs_residual_inv": float(np.median(np.abs(res_in - np.median(res_in)))) if res_in.size else 0.0,
        "corr_raw_to_colmap_inv": corr,
    }
    return a, b, inliers, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", required=True, help="split dir containing train/ (images + sparse/0)")
    ap.add_argument("--raw_dir", default="", help="DA2 raw_npy_float32 dir (default: auto-detect)")
    ap.add_argument("--out_dir", default="", help="default: <split>/train/pgdr_depth_cache_aligned")
    ap.add_argument("--min_obs", type=int, default=4, help="min triangulated obs for a per-view fit")
    args = ap.parse_args()

    import cv2
    split_root = os.path.abspath(args.split_root)
    train_dir = os.path.join(split_root, "train")
    images_dir = os.path.join(train_dir, "images")
    sparse_dir = os.path.join(train_dir, "sparse", "0")
    out_dir = args.out_dir or os.path.join(train_dir, "pgdr_depth_cache_aligned")
    report = {
        "status": "FAIL",
        "source_path": train_dir.replace("\\", "/"),
        "output_cache": os.path.abspath(out_dir).replace("\\", "/"),
        "fail_reasons": [],
        "source_kind": "raw_depth_files_aligned_to_colmap_sparse",
    }

    raw_dir = _find_raw_dir(split_root, args.raw_dir)
    if raw_dir is None:
        report["fail_reasons"].append("no DA2 raw_npy_float32 dir found (run tools/run_da2_depth.py first)")
    if not os.path.isdir(sparse_dir):
        report["fail_reasons"].append(f"train COLMAP model missing: {sparse_dir}")
    if not os.path.isdir(images_dir):
        report["fail_reasons"].append(f"train images missing: {images_dir}")
    if report["fail_reasons"]:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "pgdr_depth_build_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("[CACHE][FAIL]", "; ".join(report["fail_reasons"]))
        sys.exit(1)
    report["depth_sources_checked"] = [os.path.abspath(raw_dir).replace("\\", "/")]

    cameras, images, points3d = read_model(sparse_dir)
    img_list = sorted(images.values(), key=lambda im: _natural_lower_key(im.name))
    report["expected_train_count"] = len(img_list)
    report["train_image_names"] = [im.name for im in img_list]
    report["train_stems"] = [os.path.splitext(im.name)[0].lower() for im in img_list]

    W_t, H_t = _train_resolution(images_dir, img_list[0].name)

    # global pool for the fallback fit
    pool_raw, pool_inv = [], []
    per_view_obs = []
    raw_maps = []
    missing = []
    for im in img_list:
        stem = os.path.splitext(im.name)[0]
        npy = os.path.join(raw_dir, stem + ".npy")
        if not os.path.isfile(npy):
            missing.append(stem.lower())
            raw_maps.append(None)
            per_view_obs.append(([], []))
            continue
        raw = np.load(npy).astype(np.float32)
        raw_maps.append(raw)
        R = qvec2rotmat(im.qvec)
        t = im.tvec
        # COLMAP xys live in the COLMAP camera's pixel grid; the DA2 raw may be at a different
        # resolution (e.g. raws generated from 1600-wide images while COLMAP ran at native).
        # Sampling without this rescale silently destroys the raw<->depth pairing (measured:
        # corr collapses from +0.99 to ~0 and fits invert).
        cam = cameras[im.camera_id]
        sx = raw.shape[1] / float(cam.width)
        sy = raw.shape[0] / float(cam.height)
        rv, iv = [], []
        for xy, pid in zip(im.xys, im.point3D_ids):
            if pid == -1 or pid not in points3d:
                continue
            z = float((R @ points3d[pid].xyz + t)[2])
            if z <= 1e-6:
                continue
            s = _bilinear_sample(raw, float(xy[0]) * sx, float(xy[1]) * sy)
            if not np.isfinite(s):
                continue
            rv.append(s)
            iv.append(1.0 / z)
        per_view_obs.append((rv, iv))
        pool_raw += rv
        pool_inv += iv

    if missing:
        report["missing_depth_stems"] = missing
        report["fail_reasons"].append(f"missing raw depth for stems: {missing}")
    global_fit = _robust_fit(pool_raw, pool_inv) if len(pool_raw) >= 2 else None
    if global_fit is None:
        report["fail_reasons"].append("no usable COLMAP observations for the global fit")
    if report["fail_reasons"]:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "pgdr_depth_build_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("[CACHE][FAIL]", "; ".join(report["fail_reasons"]))
        sys.exit(1)

    frames, slices = [], []
    for idx, im in enumerate(img_list):
        raw = raw_maps[idx]
        rv, iv = per_view_obs[idx]
        fit = _robust_fit(rv, iv) if len(rv) >= args.min_obs else None
        if fit is not None and fit[0] <= 0:
            fit = None                       # degenerate negative scale -> global fallback
        if fit is not None:
            a, b, _, stats = fit
            scope = "per_view_low_support" if len(rv) < 20 else "per_view"
        else:
            a, b, _, stats = global_fit
            scope = "global_fallback"
        resized = cv2.resize(raw, (W_t, H_t), interpolation=cv2.INTER_LINEAR)
        aligned = np.clip(a * resized.astype(np.float64) + b, 0.0, None).astype(np.float32)
        slices.append(torch.from_numpy(aligned)[None])
        q_view = float(np.clip(stats["corr_raw_to_colmap_inv"], 0.0, 1.0) * stats["inlier_ratio"])
        frames.append({
            "index": idx,
            "image_name": im.name,
            "stem": os.path.splitext(im.name)[0].lower(),
            "width": W_t,
            "height": H_t,
            "raw_depth_file": os.path.abspath(os.path.join(raw_dir, os.path.splitext(im.name)[0] + ".npy")).replace("\\", "/"),
            "raw_depth_shape": [int(raw.shape[0]), int(raw.shape[1])],
            "resized": bool(raw.shape[0] != H_t or raw.shape[1] != W_t),
            "resize_mode": "bilinear",
            "fit_a": a,
            "fit_b": b,
            "fit_scope": scope,
            "q_view": q_view,
            "q_view_source": "robust_sparse_colmap_alignment",
            "num_sparse_observations": len(rv),
            "num_valid_depth_samples": len(rv),
            **stats,
            "finite_ratio": float(np.isfinite(aligned).mean()),
            "positive_ratio": float((aligned > 0).mean()),
            "inv_depth_min": float(aligned.min()),
            "inv_depth_median": float(np.median(aligned)),
            "inv_depth_max": float(aligned.max()),
            "status": "PASS",
            "fail_reasons": [],
        })

    tensor = torch.stack(slices, dim=0).contiguous()        # [V,1,H,W]
    os.makedirs(out_dir, exist_ok=True)
    torch.save(tensor, os.path.join(out_dir, "depth_inv_aligned.pt"))

    meta = {
        "status": "PASS",
        "schema": SCHEMA,
        "cache_type": "aligned_inverse_depth",
        "image_order": "natural_lowercase_image_name",
        "num_views": len(frames),
        "frames": frames,
    }
    with open(os.path.join(out_dir, "depth_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report.update({
        "status": "PASS",
        "missing_depth_stems": [],
        "num_views": len(frames),
        "shape": list(tensor.shape),
        "finite_ratio_min": float(min(fr["finite_ratio"] for fr in frames)),
        "finite_ratio_mean": float(np.mean([fr["finite_ratio"] for fr in frames])),
    })
    with open(os.path.join(out_dir, "pgdr_depth_build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    qvs = [fr["q_view"] for fr in frames]
    align_report = {
        "status": "PASS",
        "method": "per_view_robust_affine_on_colmap_sparse_inverse_depth",
        "uses_colmap_sparse": True,
        "num_views": len(frames),
        "aligned_views": len(frames),
        "global_fit_a": global_fit[0],
        "global_fit_b": global_fit[1],
        "global_num_valid_depth_samples": len(pool_raw),
        "q_view_min": float(min(qvs)),
        "q_view_median": float(sorted(qvs)[len(qvs) // 2]),
        "inlier_ratio_median": float(np.median([fr["inlier_ratio"] for fr in frames])),
        "median_abs_residual_inv_median": float(np.median([fr["median_abs_residual_inv"] for fr in frames])),
        "finite_ratio_min": float(min(fr["finite_ratio"] for fr in frames)),
        "positive_ratio_mean": float(np.mean([fr["positive_ratio"] for fr in frames])),
        "per_view": frames,
        "fail_reasons": [],
    }
    with open(os.path.join(out_dir, "sparse_alignment_report.json"), "w", encoding="utf-8") as f:
        json.dump(align_report, f, indent=2)

    print(f"[CACHE] PASS: {len(frames)} views {W_t}x{H_t} -> {out_dir}")
    print(f"[CACHE] q_view min/median = {min(qvs):.3f}/{sorted(qvs)[len(qvs) // 2]:.3f}; "
          f"scopes = {[fr['fit_scope'] for fr in frames]}")


if __name__ == "__main__":
    main()
