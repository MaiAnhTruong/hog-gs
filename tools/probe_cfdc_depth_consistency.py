"""Probe cross-fitted depth confidence for sparse-view 3DGS splits.

This is a no-training diagnostic. It checks whether the aligned monocular
inverse-depth cache is geometrically self-consistent across train views.

For a source pixel p in view v:
  1. Convert inverse depth to camera depth z.
  2. Backproject p to world.
  3. Project that 3D point into every other train view u.
  4. Compare projected camera depth with u's aligned depth at the projected pixel.

The resulting support count is a candidate confidence map for a future
Cross-Fitted Depth Confidence (CFDC) loss:
  depth loss weight high only when other train views confirm the same surface.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel, Scene
from utils.pgdr_utils import _project_points_to_camera, _unproject_pixel_to_world


def _stem(name):
    return os.path.splitext(os.path.basename(str(name)))[0].lower()


def _load_depth_cache(cameras, cache_dir, device):
    cache_dir = Path(cache_dir)
    inv = torch.load(cache_dir / "depth_inv_aligned.pt", map_location="cpu").float()
    meta = json.load(open(cache_dir / "depth_meta.json", "r", encoding="utf-8"))
    frames = meta.get("frames", [])
    order = [str(f.get("stem") or _stem(f.get("image_name", ""))).lower() for f in frames]

    out = {}
    for cam in cameras:
        s = _stem(getattr(cam, "image_name", ""))
        if s not in order:
            continue
        d = inv[order.index(s)].squeeze().to(device=device, dtype=torch.float32)
        out[s] = d
    return out, meta


def _camera_intrinsics(cam):
    h, w = int(cam.image_height), int(cam.image_width)
    fx = 0.5 * float(w) / max(math.tan(float(cam.FoVx) * 0.5), 1e-8)
    fy = 0.5 * float(h) / max(math.tan(float(cam.FoVy) * 0.5), 1e-8)
    return fx, fy, float(w) * 0.5, float(h) * 0.5


@torch.no_grad()
def probe(args):
    device = torch.device(args.device)

    # Build a minimal Scene to reuse the repository camera loader/conventions.
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ns = parser.parse_args([])
    ns.source_path = os.path.join(args.split_root, "train")
    ns.model_path = args.model_path
    ns.images = "images"
    ns.depths = ""
    ns.eval = False
    ns.resolution = args.resolution
    ns.data_device = args.device
    ns.train_test_exp = False
    ns.sh_degree = 3

    os.makedirs(args.model_path, exist_ok=True)
    gaussians = GaussianModel(3)
    scene = Scene(lp.extract(ns), gaussians, shuffle=False)
    cams = sorted(scene.getTrainCameras(), key=lambda c: _stem(c.image_name))
    depth_by_stem, meta = _load_depth_cache(cams, args.depth_cache, device)
    if len(depth_by_stem) != len(cams):
        raise RuntimeError(f"matched depth {len(depth_by_stem)}/{len(cams)} cameras")

    tau_values = [float(x) for x in args.taus.split(",")]
    stats = {
        "split_root": args.split_root,
        "depth_cache": args.depth_cache,
        "num_cameras": len(cams),
        "stride": args.stride,
        "taus": tau_values,
        "per_view": [],
        "aggregate": {},
        "depth_meta_schema": meta.get("schema"),
        "depth_meta_status": meta.get("status"),
    }

    all_counts_by_tau = {tau: [] for tau in tau_values}
    all_valid = []
    confidence_maps = []
    confidence_tau = float(args.confidence_tau)
    if confidence_tau not in tau_values:
        raise ValueError("--confidence_tau must be present in --taus")

    for cam in cams:
        s = _stem(cam.image_name)
        inv_src = depth_by_stem[s]
        h, w = inv_src.shape
        yy, xx = torch.meshgrid(
            torch.arange(0, h, args.stride, device=device, dtype=torch.float32),
            torch.arange(0, w, args.stride, device=device, dtype=torch.float32),
            indexing="ij",
        )
        x = xx.reshape(-1)
        y = yy.reshape(-1)
        inv_sample = inv_src[y.long(), x.long()]
        valid_src = torch.isfinite(inv_sample) & (inv_sample > 1e-6)
        z = 1.0 / inv_sample.clamp(min=1e-6)
        fx, fy, cx, cy = _camera_intrinsics(cam)
        pts = _unproject_pixel_to_world(cam, x, y, z, fx, fy, cx, cy, device, torch.float32)

        support = {tau: torch.zeros_like(z, dtype=torch.int16) for tau in tau_values}
        visible_other = torch.zeros_like(z, dtype=torch.int16)
        for other in cams:
            so = _stem(other.image_name)
            if so == s:
                continue
            proj = _project_points_to_camera(pts, other)
            valid = proj["valid"] & valid_src
            if not bool(valid.any()):
                continue
            inv_o = depth_by_stem[so]
            ho, wo = inv_o.shape
            col = proj["u"].round().long().clamp(0, wo - 1)
            row = proj["v"].round().long().clamp(0, ho - 1)
            inv_ref = inv_o[row, col]
            z_ref = 1.0 / inv_ref.clamp(min=1e-6)
            good_ref = valid & torch.isfinite(inv_ref) & (inv_ref > 1e-6)
            rel = (proj["z"] - z_ref).abs() / torch.maximum(proj["z"].abs(), z_ref.abs()).clamp(min=1e-6)
            visible_other += good_ref.to(torch.int16)
            for tau in tau_values:
                support[tau] += (good_ref & (rel < tau)).to(torch.int16)

        view_stats = {
            "image_name": cam.image_name,
            "samples": int(x.numel()),
            "valid_source_fraction": float(valid_src.float().mean().item()),
            "visible_other_mean": float(visible_other.float().mean().item()),
        }
        all_valid.append(valid_src.float().cpu())
        for tau in tau_values:
            cnt = support[tau].float()
            all_counts_by_tau[tau].append(cnt.cpu())
            view_stats[f"support_mean_tau_{tau}"] = float(cnt.mean().item())
            for k in (1, 2, 3, 4):
                view_stats[f"support_ge{k}_tau_{tau}"] = float((cnt >= k).float().mean().item())

        if args.write_confidence:
            hs = int(math.ceil(h / float(args.stride)))
            ws = int(math.ceil(w / float(args.stride)))
            cnt = support[confidence_tau].float().reshape(hs, ws)
            conf = (cnt / max(float(args.confidence_support), 1.0)).clamp(0.0, 1.0)
            if args.stride > 1:
                conf = F.interpolate(
                    conf[None, None],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].clamp(0.0, 1.0)
            confidence_maps.append(conf.cpu())
        stats["per_view"].append(view_stats)

    for tau in tau_values:
        cnt = torch.cat(all_counts_by_tau[tau])
        agg = {
            "support_mean": float(cnt.mean().item()),
            "support_p50": float(cnt.quantile(0.50).item()),
            "support_p90": float(cnt.quantile(0.90).item()),
        }
        for k in (1, 2, 3, 4):
            agg[f"support_ge{k}"] = float((cnt >= k).float().mean().item())
        stats["aggregate"][str(tau)] = agg
    stats["aggregate"]["valid_source_fraction"] = float(torch.cat(all_valid).mean().item())

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "cfdc_depth_consistency_probe.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if args.write_confidence:
        conf_tensor = torch.stack(confidence_maps, dim=0)[:, None].contiguous()
        conf_name = args.confidence_name
        conf_path = os.path.join(args.out_dir, conf_name)
        torch.save(conf_tensor, conf_path)
        meta = {
            "schema": "CFDC_DEPTH_CONFIDENCE_V1",
            "source_probe": out_json,
            "confidence_tensor": conf_name,
            "shape": list(conf_tensor.shape),
            "image_order": [cam.image_name for cam in cams],
            "tau": confidence_tau,
            "support_norm": int(args.confidence_support),
            "stride": int(args.stride),
            "definition": "clamp(cross_view_depth_support / support_norm, 0, 1), bilinear-upsampled if stride > 1",
        }
        with open(os.path.join(args.out_dir, "cfdc_confidence_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[CFDC] wrote confidence tensor {conf_path} shape={list(conf_tensor.shape)}")

    print(f"[CFDC] wrote {out_json}")
    print(f"[CFDC] cameras={len(cams)} stride={args.stride} samples/view~{stats['per_view'][0]['samples']}")
    for tau in tau_values:
        a = stats["aggregate"][str(tau)]
        print(
            f"[CFDC] tau={tau:.3f} mean_support={a['support_mean']:.3f} "
            f"ge1={a['support_ge1']:.3f} ge2={a['support_ge2']:.3f} "
            f"ge3={a['support_ge3']:.3f} ge4={a['support_ge4']:.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", required=True)
    ap.add_argument("--depth_cache", default="")
    ap.add_argument("--model_path", default="_cfdc_probe_tmp")
    ap.add_argument("--out_dir", default="_cfdc_probe")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--taus", default="0.03,0.05,0.10,0.15")
    ap.add_argument("--write_confidence", action="store_true")
    ap.add_argument("--confidence_tau", type=float, default=0.05)
    ap.add_argument("--confidence_support", type=int, default=3)
    ap.add_argument("--confidence_name", default="cfdc_confidence.pt")
    ap.add_argument("--resolution", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.depth_cache:
        args.depth_cache = os.path.join(args.split_root, "train", "pgdr_depth_cache_aligned")
    probe(args)


if __name__ == "__main__":
    main()
