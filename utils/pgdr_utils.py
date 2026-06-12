import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-8


def _canonical_name(name: str) -> str:
    return Path(str(name).replace("\\", "/")).name.lower()


def _stem_name(name: str) -> str:
    return Path(_canonical_name(name)).stem.lower()


def _natural_key(name: str):
    import re

    base = _canonical_name(name)
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", base)]


def _json_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def _append_line(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _maybe_write_header(path: str, header: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        _append_line(path, header)


def _coerce_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _depth_cache_names(per_view: Sequence[dict]) -> List[str]:
    names = []
    for row in per_view:
        name = row.get("image_name") or row.get("file_name") or row.get("image_stem") or ""
        if "." not in str(name) and row.get("image_stem"):
            names.append(str(row.get("image_stem")).lower())
        else:
            names.append(_canonical_name(name))
    return names


def _match_name(cache_name: str, camera_name: str) -> bool:
    if _canonical_name(cache_name) == _canonical_name(camera_name):
        return True
    return _stem_name(cache_name) == _stem_name(camera_name)


def _camera_hw(camera) -> Tuple[int, int]:
    return int(getattr(camera, "image_height", getattr(camera, "height", 0))), int(
        getattr(camera, "image_width", getattr(camera, "width", 0))
    )


def _project_points_to_camera(points_xyz: torch.Tensor, camera) -> dict:
    device = points_xyz.device
    dtype = points_xyz.dtype
    R = torch.as_tensor(camera.R, device=device, dtype=dtype)
    T = torch.as_tensor(camera.T, device=device, dtype=dtype)
    pc = points_xyz @ R + T
    z = pc[:, 2]
    in_front = z > 1e-6

    height, width = _camera_hw(camera)
    tan_x = max(math.tan(float(camera.FoVx) * 0.5), 1e-8)
    tan_y = max(math.tan(float(camera.FoVy) * 0.5), 1e-8)
    fx = 0.5 * float(width) / tan_x
    fy = 0.5 * float(height) / tan_y
    cx = float(width) * 0.5
    cy = float(height) * 0.5

    z_safe = torch.clamp(z, min=1e-6)
    u = fx * pc[:, 0] / z_safe + cx
    v = fy * pc[:, 1] / z_safe + cy
    in_image = (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
    valid = in_front & in_image

    return {
        "u": u,
        "v": v,
        "z": z,
        "valid": valid,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def _unproject_pixel_to_world(
    camera,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_cam = (x.to(dtype) - float(cx)) * z / max(float(fx), 1e-8)
    y_cam = (y.to(dtype) - float(cy)) * z / max(float(fy), 1e-8)
    p_cam = torch.stack([x_cam, y_cam, z.to(dtype)], dim=-1)
    R = torch.as_tensor(camera.R, device=device, dtype=dtype)
    T = torch.as_tensor(camera.T, device=device, dtype=dtype)
    return (p_cam - T) @ R.transpose(0, 1)


@dataclass
class PGDRConfig:
    update_interval: int = 100
    sample_views: int = 2
    pixels_per_view: int = 256
    max_candidates_per_pixel: int = 24
    support_scale: float = 1.5
    min_screen_radius: float = 1.5
    max_screen_radius: float = 64.0
    sigma_scale: float = 0.5
    alpha_scale: float = 1.0
    alpha_cap: float = 0.95
    min_pixel_contrib: float = 1e-5
    responsibility_threshold: float = 0.04
    gate_start: int = 700
    certify_min_p: float = 0.02
    certify_min_neff: float = 1.1
    certify_max_residual: float = 0.12
    pull_residual: float = 0.08
    death_front_residual: float = 0.16
    lambda_pull: float = 0.02
    lambda_death: float = 0.01


def build_pgdr_config(opt) -> PGDRConfig:
    return PGDRConfig(
        update_interval=max(1, int(getattr(opt, "pgdr_update_interval", 100))),
        sample_views=max(1, int(getattr(opt, "pgdr_sample_views", 2))),
        pixels_per_view=max(1, int(getattr(opt, "pgdr_pixels_per_view", 256))),
        max_candidates_per_pixel=max(1, int(getattr(opt, "pgdr_max_candidates_per_pixel", 24))),
        support_scale=float(getattr(opt, "pgdr_support_scale", 1.5)),
        min_screen_radius=float(getattr(opt, "pgdr_min_screen_radius", 1.5)),
        max_screen_radius=float(getattr(opt, "pgdr_max_screen_radius", 64.0)),
        sigma_scale=float(getattr(opt, "pgdr_sigma_scale", 0.5)),
        alpha_scale=float(getattr(opt, "pgdr_alpha_scale", 1.0)),
        alpha_cap=float(getattr(opt, "pgdr_alpha_cap", 0.95)),
        min_pixel_contrib=float(getattr(opt, "pgdr_min_pixel_contrib", 1e-5)),
        responsibility_threshold=float(getattr(opt, "pgdr_responsibility_threshold", 0.04)),
        gate_start=int(getattr(opt, "pgdr_gate_start", 700)),
        certify_min_p=float(getattr(opt, "pgdr_certify_min_p", 0.02)),
        certify_min_neff=float(getattr(opt, "pgdr_certify_min_neff", 1.1)),
        certify_max_residual=float(getattr(opt, "pgdr_certify_max_residual", 0.12)),
        pull_residual=float(getattr(opt, "pgdr_pull_residual", 0.08)),
        death_front_residual=float(getattr(opt, "pgdr_death_front_residual", 0.16)),
        lambda_pull=float(getattr(opt, "pgdr_lambda_pull", 0.02)),
        lambda_death=float(getattr(opt, "pgdr_lambda_death", 0.01)),
    )


def resolve_depth_cache_dir(source_path: str, depth_cache_arg: str) -> str:
    if depth_cache_arg:
        return os.path.abspath(depth_cache_arg)
    return os.path.abspath(os.path.join(source_path, "depth_da2_cache"))


def load_pgdr_depth_cache(
    depth_cache_dir: str,
    train_cameras: Sequence,
    model_path: str,
    strict: bool = True,
    device: str = "cuda",
) -> dict:
    depth_cache_dir = os.path.abspath(depth_cache_dir)
    os.makedirs(model_path, exist_ok=True)
    report_path = os.path.join(model_path, "pgdr_precheck_report.json")
    report = {
        "status": "FAIL",
        "depth_cache_dir": _json_path(depth_cache_dir),
        "depth_meta_path": _json_path(os.path.join(depth_cache_dir, "depth_meta.json")),
        "depth_tensor_path": _json_path(os.path.join(depth_cache_dir, "depth_inv_aligned.pt")),
        "train_camera_count": len(train_cameras),
        "fail_reasons": [],
    }

    meta_path = os.path.join(depth_cache_dir, "depth_meta.json")
    tensor_path = os.path.join(depth_cache_dir, "depth_inv_aligned.pt")
    if not os.path.isdir(depth_cache_dir):
        report["fail_reasons"].append(f"depth cache dir not found: {depth_cache_dir}")
    if not os.path.isfile(meta_path):
        report["fail_reasons"].append(f"depth_meta.json missing: {meta_path}")
    if not os.path.isfile(tensor_path):
        report["fail_reasons"].append(f"depth_inv_aligned.pt missing: {tensor_path}")

    meta = {}
    per_view = []
    if not report["fail_reasons"]:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            per_view = meta.get("per_view", []) or []
        except Exception as exc:
            report["fail_reasons"].append(f"failed to parse depth_meta.json: {exc}")

    cache_names = _depth_cache_names(per_view)
    sorted_cameras = sorted(train_cameras, key=lambda c: _natural_key(c.image_name))
    sorted_camera_names = [_canonical_name(c.image_name) for c in sorted_cameras]
    report["cache_image_names"] = cache_names
    report["sorted_train_camera_names"] = sorted_camera_names
    report["image_order"] = meta.get("image_order")
    report["cache_type"] = meta.get("cache_type")
    report["meta_status"] = meta.get("status")

    if meta and str(meta.get("status", "")).upper() != "PASS":
        report["fail_reasons"].append(f"depth_meta status is not PASS: {meta.get('status')}")
    if meta and str(meta.get("cache_type", "")) != "inverse_depth_aligned":
        report["fail_reasons"].append(f"depth cache type must be inverse_depth_aligned, got {meta.get('cache_type')!r}")
    if meta and str(meta.get("image_order", "")).lower() != "natural_lowercase_image_name":
        report["fail_reasons"].append(
            f"image_order must be natural_lowercase_image_name, got {meta.get('image_order')!r}"
        )
    if meta and not per_view:
        report["fail_reasons"].append("depth_meta.json has no per_view entries")
    if per_view and len(per_view) != len(train_cameras):
        report["fail_reasons"].append(
            f"depth view count mismatch: cache={len(per_view)} train_cameras={len(train_cameras)}"
        )
    if per_view and len(per_view) == len(sorted_cameras):
        mismatches = [
            {"index": i, "cache": cache_names[i], "camera": sorted_camera_names[i]}
            for i in range(len(cache_names))
            if not _match_name(cache_names[i], sorted_camera_names[i])
        ]
        if mismatches:
            report["fail_reasons"].append(f"depth/camera image order mismatch at {len(mismatches)} views")
            report["image_mismatches"] = mismatches[:32]

    q_values = [_coerce_float(row.get("q_view"), 0.0) for row in per_view]
    report["q_view_min"] = min(q_values) if q_values else 0.0
    report["q_view_median"] = sorted(q_values)[len(q_values) // 2] if q_values else 0.0
    finite_min = min((_coerce_float(row.get("raw_depth_finite_ratio"), 0.0) for row in per_view), default=0.0)
    report["finite_ratio_min"] = finite_min
    if per_view and finite_min < 0.999:
        report["fail_reasons"].append(f"finite_ratio_min below 0.999: {finite_min}")
    if per_view and report["q_view_median"] < 0.15:
        report["fail_reasons"].append(f"q_view_median below 0.15: {report['q_view_median']}")

    tensor = None
    if not report["fail_reasons"]:
        try:
            tensor = torch.load(tensor_path, map_location="cpu")
            if not torch.is_tensor(tensor):
                report["fail_reasons"].append("depth_inv_aligned.pt is not a tensor")
            elif tensor.ndim != 4 or int(tensor.shape[1]) != 1:
                report["fail_reasons"].append(f"depth tensor must have shape [V,1,H,W], got {list(tensor.shape)}")
            elif int(tensor.shape[0]) != len(train_cameras):
                report["fail_reasons"].append(
                    f"depth tensor batch mismatch: tensor={int(tensor.shape[0])} train_cameras={len(train_cameras)}"
                )
            elif not torch.isfinite(tensor).all():
                report["fail_reasons"].append("depth tensor contains non-finite values")
        except Exception as exc:
            report["fail_reasons"].append(f"failed to load depth tensor: {exc}")

    if tensor is not None and torch.is_tensor(tensor) and tensor.ndim == 4:
        cache_h, cache_w = int(tensor.shape[2]), int(tensor.shape[3])
        camera_hw = sorted(set(_camera_hw(c) for c in train_cameras))
        report["depth_tensor_shape"] = list(tensor.shape)
        report["depth_cache_resolution"] = [cache_h, cache_w]
        report["train_camera_resolutions"] = [list(hw) for hw in camera_hw]
        if len(camera_hw) != 1:
            report["fail_reasons"].append(f"train cameras have mixed resolutions: {camera_hw}")
        elif camera_hw[0] != (cache_h, cache_w):
            report["fail_reasons"].append(
                f"depth/cache resolution mismatch: cache={(cache_h, cache_w)} camera={camera_hw[0]}"
            )

    report["status"] = "PASS" if not report["fail_reasons"] else "FAIL"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[PGDR][PRECHECK] status={report['status']} report={report_path}")
    if report["status"] != "PASS":
        for reason in report["fail_reasons"]:
            print(f"[PGDR][PRECHECK][FAIL] {reason}")
        if strict:
            raise RuntimeError(f"[PGDR][ABORT] strict precheck failed. report={report_path}")
        return {"enabled": False, "report": report}

    cache_index_by_name: Dict[str, int] = {}
    for idx, name in enumerate(cache_names):
        cache_index_by_name[_canonical_name(name)] = idx
        cache_index_by_name[_stem_name(name)] = idx

    camera_to_cache = {}
    for cam in train_cameras:
        key = _canonical_name(cam.image_name)
        stem = _stem_name(cam.image_name)
        if key in cache_index_by_name:
            camera_to_cache[key] = cache_index_by_name[key]
        else:
            camera_to_cache[key] = cache_index_by_name[stem]

    tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    q_view = torch.tensor(q_values, device=tensor.device, dtype=torch.float32)
    q_view = torch.clamp(q_view, min=0.0, max=1.0)

    print(
        "[PGDR][PRECHECK] depth cache loaded: views={} resolution={}x{} q_median={:.4f}".format(
            int(tensor.shape[0]), int(tensor.shape[3]), int(tensor.shape[2]), float(report["q_view_median"])
        )
    )
    return {
        "enabled": True,
        "inv_depth": tensor,
        "q_view": q_view,
        "per_view": per_view,
        "camera_to_cache": camera_to_cache,
        "report": report,
        "report_path": report_path,
    }


class PGDRState:
    def __init__(
        self,
        cache: dict,
        train_cameras: Sequence,
        model_path: str,
        config: PGDRConfig,
    ):
        self.enabled = bool(cache.get("enabled", False))
        self.inv_depth: Optional[torch.Tensor] = cache.get("inv_depth")
        self.q_view: Optional[torch.Tensor] = cache.get("q_view")
        self.camera_to_cache = cache.get("camera_to_cache", {})
        self.train_cameras = list(train_cameras)
        self.model_path = model_path
        self.config = config
        self.device = self.inv_depth.device if self.inv_depth is not None else torch.device("cuda")
        self.count = 0
        self.last_update_iteration = 0
        self.last_summary = {}
        self.update_log_path = os.path.join(model_path, "pgdr_updates.jsonl")
        self.summary_csv_path = os.path.join(model_path, "pgdr_summary.csv")
        _maybe_write_header(
            self.summary_csv_path,
            "iteration,num_gaussians,views_sampled,pixels_sampled,pixels_used,contributors,"
            "certified,birth_hold,pull,death_candidate,mean_p,mean_residual,mean_front,mean_back,mean_neff",
        )
        self._allocate(0)

    def _allocate(self, n: int):
        self.count = int(n)
        shape = (self.count,)
        self.P = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.P_sq = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.R = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.front = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.back = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.neff = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.pull_weight = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.pull_target = torch.zeros((self.count, 3), device=self.device, dtype=torch.float32)
        self.certified = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self.birth_hold = torch.ones(shape, device=self.device, dtype=torch.bool)
        self.pull = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self.death_candidate = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self.confidence = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.mean_residual = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.mean_front = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self.mean_back = torch.zeros(shape, device=self.device, dtype=torch.float32)

    def resize_or_reset(self, n: int):
        if int(n) != self.count:
            self._allocate(int(n))
            self.last_update_iteration = 0

    def should_update(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        if iteration <= 0:
            return False
        return iteration == 1 or (iteration % self.config.update_interval == 0)

    def adc_certified_mask(self, iteration: int) -> Optional[torch.Tensor]:
        if not self.enabled or iteration < self.config.gate_start:
            return None
        return self.certified.detach().clone()

    def compute_losses(self, gaussians, iteration: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        device = gaussians.get_xyz.device
        zero = gaussians.get_xyz.sum() * 0.0
        if not self.enabled or self.count != int(gaussians.get_xyz.shape[0]):
            return zero, zero, {"pgdr_pull_count": 0, "pgdr_death_candidate_count": 0}

        pull_loss = zero
        death_loss = zero
        pull_mask = self.pull & (self.pull_weight > 0)
        death_mask = self.death_candidate

        if pull_mask.any() and self.config.lambda_pull > 0.0:
            target = self.pull_target[pull_mask].detach().to(device)
            weights = torch.clamp(self.confidence[pull_mask].detach().to(device), min=0.05, max=1.0)
            err = F.smooth_l1_loss(gaussians.get_xyz[pull_mask], target, reduction="none").sum(dim=1)
            pull_loss = (weights * err).sum() / (weights.sum() + EPS)
            pull_loss = pull_loss * float(self.config.lambda_pull)

        if death_mask.any() and self.config.lambda_death > 0.0:
            weights = torch.clamp(self.mean_front[death_mask].detach().to(device), min=0.05, max=1.0)
            opacity = gaussians.get_opacity.squeeze()[death_mask]
            death_loss = (weights * opacity).sum() / (weights.sum() + EPS)
            death_loss = death_loss * float(self.config.lambda_death)

        scalars = {
            "pgdr_pull_count": int(pull_mask.sum().item()),
            "pgdr_death_candidate_count": int(death_mask.sum().item()),
            "pgdr_certified_count": int(self.certified.sum().item()),
            "pgdr_birth_hold_count": int(self.birth_hold.sum().item()),
            "pgdr_pull_loss": float(pull_loss.detach().item()) if torch.is_tensor(pull_loss) else 0.0,
            "pgdr_death_loss": float(death_loss.detach().item()) if torch.is_tensor(death_loss) else 0.0,
        }
        return pull_loss, death_loss, scalars

    @torch.no_grad()
    def update(self, gaussians, iteration: int) -> dict:
        if not self.enabled:
            return {}
        n = int(gaussians.get_xyz.shape[0])
        self.resize_or_reset(n)
        self.P.zero_()
        self.P_sq.zero_()
        self.R.zero_()
        self.front.zero_()
        self.back.zero_()
        self.pull_weight.zero_()
        self.pull_target.zero_()

        view_count = min(self.config.sample_views, len(self.train_cameras))
        sampled_view_ids = random.sample(range(len(self.train_cameras)), view_count)
        contributors = 0
        pixels_sampled = 0
        pixels_used = 0
        dtype = gaussians.get_xyz.dtype

        for view_id in sampled_view_ids:
            camera = self.train_cameras[view_id]
            cache_idx = self.camera_to_cache[_canonical_name(camera.image_name)]
            depth_map = self.inv_depth[int(cache_idx), 0]
            valid_pixels = torch.isfinite(depth_map) & (depth_map > 1e-6)
            valid_flat = torch.nonzero(valid_pixels.flatten(), as_tuple=False).flatten()
            if valid_flat.numel() == 0:
                continue

            sample_n = min(self.config.pixels_per_view, int(valid_flat.numel()))
            perm = torch.randperm(valid_flat.numel(), device=depth_map.device)[:sample_n]
            flat = valid_flat[perm]
            ys = torch.div(flat, depth_map.shape[1], rounding_mode="floor").float()
            xs = (flat % depth_map.shape[1]).float()
            target_inv = depth_map.flatten()[flat].float()
            pixels_sampled += sample_n

            proj = _project_points_to_camera(gaussians.get_xyz.detach(), camera)
            valid = proj["valid"]
            if not valid.any():
                continue

            idx_all = torch.nonzero(valid, as_tuple=False).flatten()
            u = proj["u"][idx_all].float()
            v = proj["v"][idx_all].float()
            z = proj["z"][idx_all].float()
            opacity = gaussians.get_opacity.detach().squeeze()[idx_all].float()
            max_scale = gaussians.get_scaling.detach()[idx_all].max(dim=1).values.float()
            focal = 0.5 * (float(proj["fx"]) + float(proj["fy"]))
            radius = max_scale * focal / torch.clamp(z, min=1e-6)
            radius = torch.clamp(radius, min=self.config.min_screen_radius, max=self.config.max_screen_radius)
            sigma = torch.clamp(radius * self.config.sigma_scale, min=0.75)
            q = torch.clamp(self.q_view[int(cache_idx)], min=0.0, max=1.0).float()

            for x, y, t_inv in zip(xs, ys, target_inv):
                dx = u - x
                dy = v - y
                d2 = dx * dx + dy * dy
                support = radius * self.config.support_scale
                cand_mask = d2 <= (support * support)
                if not cand_mask.any():
                    continue

                cand_local = torch.nonzero(cand_mask, as_tuple=False).flatten()
                influence = torch.exp(-0.5 * d2[cand_local] / torch.clamp(sigma[cand_local] ** 2, min=EPS))
                alpha = torch.clamp(opacity[cand_local] * influence * self.config.alpha_scale, 0.0, self.config.alpha_cap)
                valid_alpha = alpha > 1e-5
                if not valid_alpha.any():
                    continue
                cand_local = cand_local[valid_alpha]
                influence = influence[valid_alpha]
                alpha = alpha[valid_alpha]

                if cand_local.numel() > self.config.max_candidates_per_pixel:
                    score = alpha * influence
                    _, top_idx = torch.topk(score, self.config.max_candidates_per_pixel, largest=True)
                    cand_local = cand_local[top_idx]
                    alpha = alpha[top_idx]

                order = torch.argsort(z[cand_local], descending=False)
                cand_local = cand_local[order]
                alpha = alpha[order]
                one_minus = torch.clamp(1.0 - alpha, min=EPS, max=1.0)
                trans_before = torch.cumprod(
                    torch.cat([torch.ones((1,), device=alpha.device), one_minus[:-1]], dim=0),
                    dim=0,
                )
                contrib = trans_before * alpha
                total = contrib.sum()
                if total <= self.config.min_pixel_contrib:
                    continue

                pi = contrib / (total + EPS)
                responsible = pi >= self.config.responsibility_threshold
                if not responsible.any():
                    responsible[torch.argmax(pi)] = True

                cand_local = cand_local[responsible]
                pi = pi[responsible]
                global_idx = idx_all[cand_local]
                z_i = z[cand_local]
                pred_inv = 1.0 / torch.clamp(z_i, min=1e-6)
                signed = pred_inv - t_inv
                denom = torch.clamp(t_inv.abs(), min=1e-4)
                abs_rel = signed.abs() / denom
                front_rel = torch.relu(signed) / denom
                back_rel = torch.relu(-signed) / denom
                w = pi * q

                target_z = 1.0 / torch.clamp(t_inv, min=1e-6)
                target_world = _unproject_pixel_to_world(
                    camera,
                    x.view(1),
                    y.view(1),
                    target_z.view(1),
                    proj["fx"],
                    proj["fy"],
                    proj["cx"],
                    proj["cy"],
                    device=depth_map.device,
                    dtype=dtype,
                )[0].float()
                pull_targets = target_world.view(1, 3).repeat(global_idx.numel(), 1)

                self.P.index_add_(0, global_idx, w)
                self.P_sq.index_add_(0, global_idx, w * w)
                self.R.index_add_(0, global_idx, w * abs_rel)
                self.front.index_add_(0, global_idx, w * front_rel)
                self.back.index_add_(0, global_idx, w * back_rel)
                self.pull_weight.index_add_(0, global_idx, w)
                self.pull_target.index_add_(0, global_idx, pull_targets * w[:, None])
                contributors += int(global_idx.numel())
                pixels_used += 1

        has_weight = self.P > EPS
        self.neff = (self.P * self.P) / (self.P_sq + EPS)
        self.mean_residual = torch.where(has_weight, self.R / (self.P + EPS), torch.zeros_like(self.P))
        self.mean_front = torch.where(has_weight, self.front / (self.P + EPS), torch.zeros_like(self.P))
        self.mean_back = torch.where(has_weight, self.back / (self.P + EPS), torch.zeros_like(self.P))
        self.pull_target = torch.where(
            self.pull_weight[:, None] > EPS,
            self.pull_target / (self.pull_weight[:, None] + EPS),
            gaussians.get_xyz.detach().float(),
        )
        evidence = (self.P >= self.config.certify_min_p) & (self.neff >= self.config.certify_min_neff)
        self.certified = evidence & (self.mean_residual <= self.config.certify_max_residual)
        self.birth_hold = ~evidence
        self.pull = evidence & (~self.certified) & (self.mean_residual >= self.config.pull_residual)
        self.death_candidate = (
            evidence
            & (self.mean_front >= self.config.death_front_residual)
            & (self.mean_front >= self.mean_back * 1.5)
        )
        self.confidence = torch.clamp(self.P / (self.P + 1.0), 0.0, 1.0) * torch.clamp(self.neff / 2.0, 0.0, 1.0)
        self.last_update_iteration = int(iteration)

        active = has_weight
        summary = {
            "iteration": int(iteration),
            "num_gaussians": int(n),
            "views_sampled": int(view_count),
            "pixels_sampled": int(pixels_sampled),
            "pixels_used": int(pixels_used),
            "contributors": int(contributors),
            "certified": int(self.certified.sum().item()),
            "birth_hold": int(self.birth_hold.sum().item()),
            "pull": int(self.pull.sum().item()),
            "death_candidate": int(self.death_candidate.sum().item()),
            "mean_p": float(self.P[active].mean().item()) if active.any() else 0.0,
            "mean_residual": float(self.mean_residual[active].mean().item()) if active.any() else 0.0,
            "mean_front": float(self.mean_front[active].mean().item()) if active.any() else 0.0,
            "mean_back": float(self.mean_back[active].mean().item()) if active.any() else 0.0,
            "mean_neff": float(self.neff[active].mean().item()) if active.any() else 0.0,
        }
        self.last_summary = summary
        self._log_summary(summary)
        print(
            "[PGDR][ITER {iteration}] views={views_sampled} pixels={pixels_used}/{pixels_sampled} "
            "certified={certified} birth_hold={birth_hold} pull={pull} death={death_candidate} "
            "mean_res={mean_residual:.5f}".format(**summary)
        )
        return summary

    def _log_summary(self, summary: dict):
        _append_line(self.update_log_path, json.dumps(summary, sort_keys=True))
        row = [
            summary["iteration"],
            summary["num_gaussians"],
            summary["views_sampled"],
            summary["pixels_sampled"],
            summary["pixels_used"],
            summary["contributors"],
            summary["certified"],
            summary["birth_hold"],
            summary["pull"],
            summary["death_candidate"],
            f"{summary['mean_p']:.8f}",
            f"{summary['mean_residual']:.8f}",
            f"{summary['mean_front']:.8f}",
            f"{summary['mean_back']:.8f}",
            f"{summary['mean_neff']:.8f}",
        ]
        _append_line(self.summary_csv_path, ",".join(map(str, row)))
