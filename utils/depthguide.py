#
# Depth-Anchored Densification (DAD) for sparse-view 3DGS.
#
# Research gap attacked: existing depth methods (FSGS Pearson, DNGaussian affine, SparseGS) use depth
# as a SOFT, UNIFORM rendered-depth LOSS that competes with the photometric term and loses exactly in
# the poorly-observed regions where geometry is wrong (and over-smooths elsewhere). DAD instead drives
# the Gaussian PROCESS: when a Gaussian is split, its children are seeded at the multi-view-consistent
# DEPTH-PRIOR position (not a random covariance offset), so new capacity lands on the true surface
# instead of spreading into empty space (the structural-error source). No increase of INITIAL points;
# this is densification-time placement only, gated by multi-view depth agreement (skip where the
# monocular depth is unreliable, so it never corrupts geometry it cannot trust).
#
# Target along the pixel ray: a Gaussian at X seen by camera v (center C_v) has camera depth z_v; the
# prior surface depth there is z*_v = 1/invdepth_prior. Moving X along the SAME ray to depth z*_v gives
#   X*_v = C_v + (z*_v / z_v) (X - C_v)            (camera depth is linear along the ray through C_v)
# DAD averages X*_v over the views that see X and keeps it only if the per-view depths agree.
#
import os
import json
import torch
from utils.pgdr_utils import _project_points_to_camera


@torch.no_grad()
def load_aligned_depth_cache(cameras, cache_dir):
    """Load the split's aligned inverse-depth cache (depth_inv_aligned.pt [V,1,H,W] + depth_meta.json,
    schema PGDR_DEPTH_CACHE_V2_ALIGNED, COLMAP-scale inverse depth) into camera.invdepthmap / depth_mask
    / depth_reliable, matched by image stem. Enables both the depth-reg loss and DAD. Returns #matched."""
    import cv2
    pt = os.path.join(cache_dir, "depth_inv_aligned.pt")
    meta_p = os.path.join(cache_dir, "depth_meta.json")
    if not (os.path.isfile(pt) and os.path.isfile(meta_p)):
        raise FileNotFoundError(f"[DAD] aligned depth cache not found at {cache_dir}")
    inv = torch.load(pt, map_location="cpu").float()                 # [V,1,H,W] or [V,H,W]
    meta = json.load(open(meta_p, "r"))
    frames = meta.get("frames", [])
    order = [str(f.get("stem") or os.path.splitext(str(f.get("image_name", "")))[0]).lower() for f in frames]
    matched = 0
    for cam in cameras:
        stem = os.path.splitext(os.path.basename(getattr(cam, "image_name", "")))[0].lower()
        if stem not in order:
            cam.depth_reliable = False
            cam.invdepthmap = None
            continue
        d = inv[order.index(stem)].squeeze().float()                  # [H,W]
        H, W = int(cam.image_height), int(cam.image_width)
        if d.shape[-1] != W or d.shape[-2] != H:
            d = torch.from_numpy(cv2.resize(d.numpy(), (W, H), interpolation=cv2.INTER_NEAREST))
        d = d.clamp(min=0).cuda()
        cam.invdepthmap = d
        cam.depth_mask = (d > 0).float()
        cam.depth_reliable = True
        matched += 1
    return matched


@torch.no_grad()
def load_cfdc_confidence_cache(cameras, cache_path):
    """Load CFDC cross-view depth confidence into camera.depth_confidence.

    `cache_path` may be either a directory containing cfdc_confidence.pt and
    cfdc_confidence_meta.json, or the .pt tensor itself. The tensor is [V,1,H,W]
    or [V,H,W], matched by image stem when metadata provides image_order.
    """
    import cv2
    if os.path.isdir(cache_path):
        pt = os.path.join(cache_path, "cfdc_confidence.pt")
        meta_p = os.path.join(cache_path, "cfdc_confidence_meta.json")
    else:
        pt = cache_path
        meta_p = os.path.join(os.path.dirname(cache_path), "cfdc_confidence_meta.json")
    if not os.path.isfile(pt):
        raise FileNotFoundError(f"[CFDC] confidence tensor not found: {pt}")
    conf = torch.load(pt, map_location="cpu").float()
    meta = {}
    if os.path.isfile(meta_p):
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
    order_raw = meta.get("image_order") or []
    if order_raw:
        order = [os.path.splitext(os.path.basename(str(x)))[0].lower() for x in order_raw]
    else:
        order = sorted([os.path.splitext(os.path.basename(getattr(c, "image_name", "")))[0].lower() for c in cameras])
    matched = 0
    for cam in cameras:
        stem = os.path.splitext(os.path.basename(getattr(cam, "image_name", "")))[0].lower()
        if stem not in order:
            continue
        d = conf[order.index(stem)].squeeze().float().clamp(0.0, 1.0)
        H, W = int(cam.image_height), int(cam.image_width)
        if d.shape[-1] != W or d.shape[-2] != H:
            d = torch.from_numpy(cv2.resize(d.numpy(), (W, H), interpolation=cv2.INTER_LINEAR))
        cam.depth_confidence = d[None].cuda()
        matched += 1
    return matched


@torch.no_grad()
def compute_depth_targets(gaussians, cameras, min_views=2, agree_rel=0.15, t_clip=(0.5, 2.0)):
    """Return (target_xyz [N,3], reliable [N] bool). target_xyz = depth-prior-consistent position;
    reliable = seen by >= min_views with consistent per-view target depth (relative std < agree_rel)."""
    xyz = gaussians.get_xyz.detach()
    N = xyz.shape[0]
    device = xyz.device
    sum_tgt = torch.zeros(N, 3, device=device)
    sum_t = torch.zeros(N, device=device)
    sum_t2 = torch.zeros(N, device=device)
    cnt = torch.zeros(N, device=device)

    for cam in cameras:
        if not bool(getattr(cam, "depth_reliable", False)) or getattr(cam, "invdepthmap", None) is None:
            continue
        inv = cam.invdepthmap
        inv = inv if torch.is_tensor(inv) else torch.as_tensor(inv)
        inv = inv.to(device).float().squeeze()                      # [H,W] aligned inverse depth
        if inv.dim() != 2:
            continue
        H, W = inv.shape[0], inv.shape[1]
        proj = _project_points_to_camera(xyz, cam)
        valid = proj["valid"]
        z = proj["z"]
        col = proj["u"].round().long().clamp(0, W - 1)
        row = proj["v"].round().long().clamp(0, H - 1)
        d_prior = inv[row, col]                                      # prior inverse depth at the pixel
        good = valid & torch.isfinite(d_prior) & (d_prior > 1e-6) & (z > 1e-6)
        if not bool(good.any()):
            continue
        z_target = 1.0 / d_prior.clamp(min=1e-6)                     # prior surface camera-depth
        t = (z_target / z.clamp(min=1e-6)).clamp(min=t_clip[0], max=t_clip[1])   # along-ray scale
        C = cam.camera_center.to(device)
        tgt = C[None, :] + t[:, None] * (xyz - C[None, :])          # X*_v
        w = good.float()
        sum_tgt += w[:, None] * tgt
        sum_t += w * t
        sum_t2 += w * t * t
        cnt += w

    cnt_safe = cnt.clamp(min=1.0)
    target_xyz = sum_tgt / cnt_safe[:, None]
    mean_t = sum_t / cnt_safe
    var_t = (sum_t2 / cnt_safe - mean_t * mean_t).clamp(min=0.0)
    rel_std = var_t.sqrt() / mean_t.abs().clamp(min=1e-6)
    reliable = (cnt >= float(min_views)) & (rel_std < float(agree_rel))
    # where not reliable, fall back to the original position (no correction)
    target_xyz = torch.where(reliable[:, None], target_xyz, xyz)
    return target_xyz, reliable


@torch.no_grad()
def compute_surface_witness_scores(
    gaussians,
    cameras,
    tau=0.05,
    conf_threshold=0.5,
    min_seen=2,
    front_ratio_threshold=0.5,
):
    """Return per-Gaussian surface witness scores from CFDC-confirmed depth.

    This targets RGB-fitting floaters: a Gaussian can help train RGB, but if its
    center repeatedly lies in front of a cross-view-confirmed depth surface, it
    has weak geometric witness. Scores are detached and intended to modulate
    opacity/densification softly, not to hard-move positions.

    Returns dict with [N] tensors:
      evidence: enough high-confidence depth observations
      witness: fraction of high-confidence observations on the surface shell
      front_ratio: fraction clearly in front of the surface
      back_ratio: fraction clearly behind the surface
      contradiction: evidence & front_ratio >= threshold
      conf_seen: count of high-confidence observations
    """
    xyz = gaussians.get_xyz.detach()
    n = xyz.shape[0]
    device = xyz.device
    conf_seen = torch.zeros(n, device=device)
    on_surface = torch.zeros(n, device=device)
    front = torch.zeros(n, device=device)
    back = torch.zeros(n, device=device)
    rel_sum = torch.zeros(n, device=device)

    for cam in cameras:
        if not bool(getattr(cam, "depth_reliable", False)):
            continue
        if getattr(cam, "invdepthmap", None) is None or getattr(cam, "depth_confidence", None) is None:
            continue
        inv = cam.invdepthmap
        inv = inv if torch.is_tensor(inv) else torch.as_tensor(inv)
        inv = inv.to(device).float().squeeze()
        conf = cam.depth_confidence
        conf = conf if torch.is_tensor(conf) else torch.as_tensor(conf)
        conf = conf.to(device).float().squeeze()
        if inv.dim() != 2 or conf.dim() != 2:
            continue
        h, w = inv.shape
        proj = _project_points_to_camera(xyz, cam)
        valid = proj["valid"]
        col = proj["u"].round().long().clamp(0, w - 1)
        row = proj["v"].round().long().clamp(0, h - 1)
        inv_ref = inv[row, col]
        z_ref = 1.0 / inv_ref.clamp(min=1e-6)
        cv = conf[row, col].clamp(0.0, 1.0)
        good = valid & torch.isfinite(inv_ref) & (inv_ref > 1e-6) & (cv >= float(conf_threshold))
        if not bool(good.any()):
            continue
        z = proj["z"]
        rel = (z - z_ref).abs() / torch.maximum(z.abs(), z_ref.abs()).clamp(min=1e-6)
        conf_seen += good.float()
        on_surface += (good & (rel <= float(tau))).float()
        front += (good & (z < z_ref * (1.0 - float(tau)))).float()
        back += (good & (z > z_ref * (1.0 + float(tau)))).float()
        rel_sum += good.float() * rel.clamp(max=10.0)

    denom = conf_seen.clamp(min=1.0)
    evidence = conf_seen >= float(min_seen)
    witness = on_surface / denom
    front_ratio = front / denom
    back_ratio = back / denom
    contradiction = evidence & (front_ratio >= float(front_ratio_threshold))
    return {
        "evidence": evidence,
        "witness": witness,
        "front_ratio": front_ratio,
        "back_ratio": back_ratio,
        "contradiction": contradiction,
        "conf_seen": conf_seen,
        "rel_mean": rel_sum / denom,
    }
