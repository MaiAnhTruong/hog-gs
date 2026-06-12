"""Probe Surface-Witnessed Depth (SWD) scores for trained Gaussians.

This is a no-training diagnostic aimed at the failure mode where floaters fit all
training RGB views. It uses model-independent depth confidence maps to ask
whether each Gaussian center lies on a cross-view-confirmed surface.

For each Gaussian center X_i and train camera v:
  - project X_i to pixel p_v and camera depth z_i_v;
  - sample aligned inverse-depth z*_v and CFDC confidence c_v at p_v;
  - if c_v is high, count whether X_i is on the surface, in front of it, or behind it.

The resulting front-violation score targets floaters in empty space even if they
help reduce train RGB loss.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel, Scene
from train import _load_checkpoint_compat
from utils.depthguide import load_aligned_depth_cache, load_cfdc_confidence_cache
from utils.pgdr_utils import _project_points_to_camera


def _stem(name):
    return os.path.splitext(os.path.basename(str(name)))[0].lower()


def _safe_quantiles(x, qs=(0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)):
    if x.numel() == 0:
        return {str(q): None for q in qs}
    if x.numel() > 2_000_000:
        x = x[:: max(1, x.numel() // 2_000_000)].contiguous()
    vals = torch.quantile(x.float().cpu(), torch.tensor(qs))
    return {str(q): float(v) for q, v in zip(qs, vals)}


@torch.no_grad()
def probe(args):
    device = torch.device(args.device)
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
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = lp.extract(ns)
    opt = op.extract(ns)
    gaussians = GaussianModel(3, opt.optimizer_type)
    try:
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    except FileNotFoundError as exc:
        checkpoint_path = os.path.join(args.model_path, f"chkpnt{int(args.iteration)}.pth")
        if not os.path.exists(checkpoint_path):
            raise exc
        scene = Scene(dataset, gaussians, shuffle=False)
        model_params, ckpt_iter = _load_checkpoint_compat(checkpoint_path)
        if int(ckpt_iter) != int(args.iteration):
            print(f"[SWD][WARN] checkpoint reports iteration {ckpt_iter}, requested {args.iteration}")
        gaussians.restore(model_params, opt)
    cams = sorted(scene.getTrainCameras(), key=lambda c: _stem(c.image_name))
    dmatched = load_aligned_depth_cache(cams, args.depth_cache)
    cmatched = load_cfdc_confidence_cache(cams, args.cfdc_cache)
    if dmatched != len(cams) or cmatched != len(cams):
        raise RuntimeError(f"depth/confidence mismatch: depth={dmatched} cfdc={cmatched} cams={len(cams)}")

    xyz = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach().squeeze()
    max_scale = gaussians.get_scaling.detach().max(dim=1).values
    n = xyz.shape[0]
    conf_seen = torch.zeros(n, device=device)
    on_surface = torch.zeros(n, device=device)
    front = torch.zeros(n, device=device)
    back = torch.zeros(n, device=device)
    rel_sum = torch.zeros(n, device=device)
    conf_sum = torch.zeros(n, device=device)

    for cam in cams:
        proj = _project_points_to_camera(xyz, cam)
        valid = proj["valid"]
        if not bool(valid.any()):
            continue
        inv = cam.invdepthmap.cuda().float().squeeze()
        conf = cam.depth_confidence.cuda().float().squeeze()
        h, w = inv.shape
        col = proj["u"].round().long().clamp(0, w - 1)
        row = proj["v"].round().long().clamp(0, h - 1)
        inv_ref = inv[row, col]
        z_ref = 1.0 / inv_ref.clamp(min=1e-6)
        cv = conf[row, col].clamp(0.0, 1.0)
        good = valid & torch.isfinite(inv_ref) & (inv_ref > 1e-6) & (cv >= float(args.conf_threshold))
        if not bool(good.any()):
            continue
        z = proj["z"]
        rel = (z - z_ref).abs() / torch.maximum(z.abs(), z_ref.abs()).clamp(min=1e-6)
        conf_seen += good.float()
        on_surface += (good & (rel <= float(args.tau))).float()
        front += (good & (z < z_ref * (1.0 - float(args.tau)))).float()
        back += (good & (z > z_ref * (1.0 + float(args.tau)))).float()
        rel_sum += good.float() * rel.clamp(max=10.0)
        conf_sum += valid.float() * cv

    denom = conf_seen.clamp(min=1.0)
    witness = on_surface / denom
    front_ratio = front / denom
    back_ratio = back / denom
    rel_mean = rel_sum / denom
    has_evidence = conf_seen >= float(args.min_seen)
    contradiction = has_evidence & (front_ratio >= float(args.front_ratio))

    # Floater proxy: isolation to a random anchor subset. This is not a label, only a sanity check.
    g = torch.Generator(device=device).manual_seed(int(args.seed))
    anchor_count = min(int(args.anchor_count), n)
    anchors = xyz[torch.randperm(n, generator=g, device=device)[:anchor_count]]
    isolation = torch.empty(n, device=device)
    for i in range(0, n, int(args.batch)):
        isolation[i:i + int(args.batch)] = torch.cdist(xyz[i:i + int(args.batch)], anchors).min(dim=1).values

    def group_stats(mask):
        mask = mask.bool()
        if not bool(mask.any()):
            return {"count": 0}
        return {
            "count": int(mask.sum().item()),
            "fraction": float(mask.float().mean().item()),
            "witness_mean": float(witness[mask].mean().item()),
            "front_ratio_mean": float(front_ratio[mask].mean().item()),
            "back_ratio_mean": float(back_ratio[mask].mean().item()),
            "conf_seen_mean": float(conf_seen[mask].mean().item()),
            "rel_mean": float(rel_mean[mask].mean().item()),
            "opacity_mean": float(opacity[mask].mean().item()),
            "max_scale_mean": float(max_scale[mask].mean().item()),
            "isolation_mean": float(isolation[mask].mean().item()),
        }

    hi_iso = isolation >= torch.quantile(isolation.float().cpu(), 0.90).to(device)
    hi_front = front_ratio >= torch.quantile(front_ratio.float().cpu(), 0.90).to(device)
    low_witness = has_evidence & (witness <= float(args.low_witness))

    report = {
        "split_root": args.split_root,
        "model_path": args.model_path,
        "iteration": int(args.iteration),
        "gaussians": int(n),
        "depth_cache": args.depth_cache,
        "cfdc_cache": args.cfdc_cache,
        "tau": float(args.tau),
        "conf_threshold": float(args.conf_threshold),
        "min_seen": float(args.min_seen),
        "front_ratio_gate": float(args.front_ratio),
        "aggregate": {
            "has_evidence_fraction": float(has_evidence.float().mean().item()),
            "contradiction_fraction_all": float(contradiction.float().mean().item()),
            "contradiction_fraction_evidence": float((contradiction & has_evidence).float().sum().item() / has_evidence.float().sum().clamp(min=1.0).item()),
            "conf_seen": _safe_quantiles(conf_seen),
            "witness": _safe_quantiles(witness[has_evidence]),
            "front_ratio": _safe_quantiles(front_ratio[has_evidence]),
            "back_ratio": _safe_quantiles(back_ratio[has_evidence]),
            "rel_mean": _safe_quantiles(rel_mean[has_evidence]),
        },
        "groups": {
            "all": group_stats(torch.ones(n, dtype=torch.bool, device=device)),
            "has_evidence": group_stats(has_evidence),
            "contradiction": group_stats(contradiction),
            "low_witness": group_stats(low_witness),
            "high_isolation_top10": group_stats(hi_iso),
            "high_front_top10": group_stats(hi_front),
            "high_isolation_and_contradiction": group_stats(hi_iso & contradiction),
        },
    }

    out_json = os.path.join(args.out_dir, f"swd_surface_witness_iter{int(args.iteration)}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SWD] wrote {out_json}")
    print(f"[SWD] N={n} evidence={report['aggregate']['has_evidence_fraction']:.3f} "
          f"contradict_all={report['aggregate']['contradiction_fraction_all']:.3f} "
          f"contradict_evidence={report['aggregate']['contradiction_fraction_evidence']:.3f}")
    for name in ("contradiction", "low_witness", "high_isolation_top10", "high_isolation_and_contradiction"):
        gstat = report["groups"][name]
        print(f"[SWD] {name}: count={gstat.get('count')} frac={gstat.get('fraction')} "
              f"front={gstat.get('front_ratio_mean')} iso={gstat.get('isolation_mean')} "
              f"opacity={gstat.get('opacity_mean')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--iteration", type=int, default=7000)
    ap.add_argument("--depth_cache", required=True)
    ap.add_argument("--cfdc_cache", required=True)
    ap.add_argument("--out_dir", default="_swd_probe")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--conf_threshold", type=float, default=0.5)
    ap.add_argument("--min_seen", type=float, default=2)
    ap.add_argument("--front_ratio", type=float, default=0.5)
    ap.add_argument("--low_witness", type=float, default=0.25)
    ap.add_argument("--anchor_count", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resolution", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    probe(ap.parse_args())


if __name__ == "__main__":
    main()
