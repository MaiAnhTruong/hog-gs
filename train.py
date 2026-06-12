#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import math
import torch
from random import randint
from datetime import datetime
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.pgdr_utils import PGDRState, build_pgdr_config, load_pgdr_depth_cache, resolve_depth_cache_dir
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _default_auto_split_root(source_path):
    scene_name = os.path.basename(os.path.normpath(source_path))
    source_abs = os.path.abspath(source_path)
    current = source_abs
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            break
        if os.path.basename(parent).lower() == "data":
            return os.path.join(parent, "_3dgs_splits", scene_name)
        current = parent
    return os.path.join(os.path.dirname(source_abs), "_3dgs_splits", scene_name)


def _apply_sparse12_10k_preset(args):
    if not args.sparse12_10k:
        return

    args.eval = True
    args.disable_viewer = True
    args.iterations = 10_000
    args.test_iterations = list(range(1000, 10001, 1000))
    args.save_iterations = [7000, 10000]
    args.checkpoint_iterations = [7000, 10000]
    args.metrics_log_interval = 1000
    args.metrics_eval_train_count = -1
    args.metrics_eval_per_view = True
    args.split_train_views = "12"
    args.split_hold = 8
    args.split_copy_mode = "copy"
    args.split_force = True
    args.split_init_policy = "sparsegs_triangulate"
    args.split_train_sample_mode = "paper_even"
    if not args.split_output_root:
        args.split_output_root = _default_auto_split_root(args.source_path)

    print("[PRESET] sparse12_10k enabled")
    print(f"[PRESET] split_output_root={args.split_output_root}")


def _as_float(x):
    if isinstance(x, float):
        return x
    if isinstance(x, int):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().mean().item())
    return float(x)


def _dropgaussian_rate(opt, iteration):
    if not getattr(opt, "dropgaussian_enable", False):
        return 0.0

    start = int(getattr(opt, "dropgaussian_start", 0))
    end = int(getattr(opt, "dropgaussian_end", opt.iterations))
    max_rate = float(getattr(opt, "dropgaussian_max_rate", 0.2))
    schedule = str(getattr(opt, "dropgaussian_schedule", "linear")).lower()
    if not 0.0 <= max_rate < 1.0:
        raise ValueError("--dropgaussian_max_rate must be in [0, 1)")
    if iteration < start:
        return 0.0
    if schedule == "constant" or end <= start:
        return max_rate

    progress = min(max((iteration - start) / float(end - start), 0.0), 1.0)
    if schedule == "linear":
        return max_rate * progress
    if schedule == "cosine":
        return max_rate * 0.5 * (1.0 - math.cos(math.pi * progress))
    raise ValueError("Unsupported --dropgaussian_schedule: {}".format(schedule))


def _safe_metric_value(x):
    try:
        value = _as_float(x)
        if math.isfinite(value):
            return value
        return float("nan")
    except Exception:
        return float("nan")


def _finite_or_raise(name, value, iteration, split, image_name=None):
    if value is None:
        return None

    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(
            f"[METRICS] Non-finite metric {name}={value} "
            f"at iteration={iteration}, split={split}, image={image_name}"
        )

    return value


def _init_lpips_model(args, device="cuda"):
    """
    Initialize LPIPS only when explicitly requested.
    Baseline training must not depend on LPIPS unless metrics flag is enabled.
    """
    if not getattr(args, "metrics_compute_lpips", False):
        return None, "disabled"

    try:
        import lpips
    except Exception as e:
        raise RuntimeError(
            "[METRICS][LPIPS] --metrics_compute_lpips was enabled, "
            "but package 'lpips' is not available. Install it with: "
            "python -m pip install lpips"
        ) from e

    model = lpips.LPIPS(net="vgg")
    model = model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, "enabled"


def _compute_lpips_value(lpips_model, image, gt):
    """
    image, gt: torch.Tensor [3,H,W], range [0,1]
    return float
    """
    if lpips_model is None:
        return None

    with torch.no_grad():
        pred = torch.clamp(image, 0.0, 1.0).unsqueeze(0).float()
        target = torch.clamp(gt, 0.0, 1.0).unsqueeze(0).float()

        pred = pred * 2.0 - 1.0
        target = target * 2.0 - 1.0

        val = lpips_model(pred, target)

        if isinstance(val, torch.Tensor):
            val = val.mean()

        if not torch.isfinite(val):
            raise RuntimeError(
                "[METRICS][LPIPS] Non-finite LPIPS value detected. "
                f"image shape={tuple(image.shape)}, gt shape={tuple(gt.shape)}, "
                f"image min/max={float(image.min())}/{float(image.max())}, "
                f"gt min/max={float(gt.min())}/{float(gt.max())}"
            )

        return float(val.item())


def _cuda_elapsed_ms(start_event, end_event):
    try:
        return start_event.elapsed_time(end_event)
    except RuntimeError as exc:
        if "not ready" not in str(exc).lower():
            raise
        end_event.synchronize()
        return start_event.elapsed_time(end_event)


def _format_float(x, digits=8):
    if x is None:
        return "nan"
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return "nan"
        return f"{x:.{digits}f}"
    except Exception:
        return "nan"


def _format_optional_metric(value, status, digits=6):
    if value is None:
        return status
    return _format_float(value, digits)


def _append_line(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _read_image_name_manifest(path):
    if not path:
        return []
    names = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Metric evaluation manifest contains duplicate image names: {}".format(path))
    return names


def _select_manifest_cameras(cameras, manifest_path):
    names = _read_image_name_manifest(manifest_path)
    if not names:
        return list(cameras)

    by_name = {}
    for camera in cameras:
        name = getattr(camera, "image_name", "")
        if name in by_name:
            raise ValueError("Duplicate test camera image_name: {}".format(name))
        by_name[name] = camera

    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(
            "Metric evaluation manifest contains names absent from the test split: {}".format(
                ", ".join(missing)
            )
        )
    return [by_name[name] for name in names]


def _load_checkpoint_compat(path):
    def install_numpy_core_aliases():
        import importlib
        import numpy as np

        sys.modules.setdefault("numpy._core", np.core)
        for name in ("multiarray", "_multiarray_umath", "numeric", "fromnumeric"):
            try:
                module = importlib.import_module("numpy.core.{}".format(name))
                sys.modules.setdefault("numpy._core.{}".format(name), module)
            except Exception:
                pass

    try:
        return torch.load(path)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy._core":
            raise
        install_numpy_core_aliases()
        return torch.load(path)
    except SystemError as exc:
        if "structseq" not in str(exc):
            raise
        install_numpy_core_aliases()
        return torch.load(path)


def _maybe_write_header(path, header):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        _append_line(path, header)


class TrainMetricsFileLogger:
    def __init__(
        self,
        model_path,
        train_count=0,
        test_count=0,
        split_report_path=None,
        validation_report_path=None,
        source_path_original=None,
        source_path_after_split=None,
        external_test_source_path=None,
        total_iterations=None,
        metrics_compute_lpips=False,
    ):
        self.model_path = model_path
        self.train_count = int(train_count)
        self.test_count = int(test_count)
        self.split_report_path = split_report_path
        self.validation_report_path = validation_report_path
        self.source_path_original = source_path_original
        self.source_path_after_split = source_path_after_split
        self.external_test_source_path = external_test_source_path
        self.total_iterations = total_iterations
        self.metrics_compute_lpips = bool(metrics_compute_lpips)
        self.history_path = os.path.join(model_path, "metrics_history.jsonl")
        self.summary_path = os.path.join(model_path, "metrics_summary.csv")
        self.readable_path = os.path.join(model_path, "metrics_readable.txt")
        self.metrics_dir = os.path.join(model_path, "metrics")
        self.metrics_summary_json_path = os.path.join(self.metrics_dir, "metrics_summary.json")
        self.metrics_summary_tsv_path = os.path.join(self.metrics_dir, "metrics_summary.tsv")
        self.run_manifest_path = os.path.join(self.metrics_dir, "run_manifest.json")
        self.summary_records = []
        self.metrics_by_iteration = {}

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        summary_fields = [
            "iteration",
            "split",
            "num_views",
            "num_gaussians",
            "l1",
            "mse",
            "rmse",
            "psnr",
            "ssim",
            "lpips",
            "lpips_status",
            "train_count",
            "test_count",
            "split_report",
        ]
        _maybe_write_header(self.summary_path, ",".join(summary_fields))
        _maybe_write_header(self.metrics_summary_tsv_path, "\t".join(summary_fields))

    def _json_metric(self, value):
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    def _csv_metric(self, value, digits=8):
        value = self._json_metric(value)
        if value is None:
            return ""
        return f"{value:.{digits}f}"

    def _quote_csv(self, value):
        text = "" if value is None else str(value)
        return '"' + text.replace('"', '""') + '"'

    def _summary_record(self, iteration, timestamp, split_name, summary):
        lpips_value = summary.get("lpips")
        return {
            "iteration": int(iteration),
            "split": split_name,
            "num_views": int(summary.get("num_views", summary.get("n_images", 0))),
            "num_gaussians": int(summary.get("num_gaussians", 0)),
            "l1": self._json_metric(summary.get("l1")),
            "mse": self._json_metric(summary.get("mse")),
            "rmse": self._json_metric(summary.get("rmse")),
            "psnr": self._json_metric(summary.get("psnr")),
            "ssim": self._json_metric(summary.get("ssim")),
            "lpips": self._json_metric(lpips_value),
            "lpips_status": summary.get("lpips_status", "disabled" if self._json_metric(lpips_value) is None else "enabled"),
            "train_count": self.train_count,
            "test_count": self.test_count,
            "split_report": self.split_report_path,
        }

    def _write_summary_row(self, record):
        row = [
            record["iteration"],
            self._quote_csv(record["split"]),
            record["num_views"],
            record["num_gaussians"],
            self._csv_metric(record["l1"]),
            self._csv_metric(record["mse"]),
            self._csv_metric(record["rmse"]),
            self._csv_metric(record["psnr"]),
            self._csv_metric(record["ssim"]),
            self._csv_metric(record["lpips"]),
            self._quote_csv(record["lpips_status"]),
            record["train_count"],
            record["test_count"],
            self._quote_csv(record["split_report"]),
        ]
        _append_line(self.summary_path, ",".join(map(str, row)))

    def _write_summary_tsv_row(self, record):
        row = [
            record["iteration"],
            record["split"],
            record["num_views"],
            record["num_gaussians"],
            self._csv_metric(record["l1"]),
            self._csv_metric(record["mse"]),
            self._csv_metric(record["rmse"]),
            self._csv_metric(record["psnr"]),
            self._csv_metric(record["ssim"]),
            self._csv_metric(record["lpips"]),
            record["lpips_status"],
            record["train_count"],
            record["test_count"],
            record["split_report"] or "",
        ]
        _append_line(self.metrics_summary_tsv_path, "\t".join(map(str, row)))

    def _write_metrics_summary_json(self):
        split_meta = self._split_report_meta()
        doc = {
            "source_path_original": self.source_path_original,
            "source_path_after_split": self.source_path_after_split,
            "external_test_source_path": self.external_test_source_path,
            "split_report_path": self.split_report_path,
            "validation_report_path": self.validation_report_path,
            "split_protocol": split_meta.get("protocol"),
            "split_init_policy": split_meta.get("split_init_policy"),
            "train_camera_count": self.train_count,
            "test_camera_count": self.test_count,
            "records": self.summary_records,
        }
        with open(self.metrics_summary_json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, allow_nan=False)

    def _split_report_meta(self):
        if not self.split_report_path or not os.path.isfile(self.split_report_path):
            return {}
        try:
            with open(self.split_report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            return {
                "protocol": report.get("protocol"),
                "split_init_policy": report.get("split_init_policy"),
                "split_train_views": report.get("split_train_views"),
                "split_hold": report.get("split_hold"),
            }
        except Exception:
            return {}

    def _update_run_manifest(self, iteration, eval_summaries):
        split_meta = self._split_report_meta()
        entry = {}
        for split_name, summary in eval_summaries.items():
            prefix = "test" if split_name == "test" else split_name
            entry[f"{prefix}_psnr_mean"] = self._json_metric(summary.get("psnr"))
            entry[f"{prefix}_ssim_mean"] = self._json_metric(summary.get("ssim"))
            entry[f"{prefix}_lpips_mean"] = self._json_metric(summary.get("lpips"))
            entry[f"{prefix}_lpips_status"] = summary.get("lpips_status")
            entry[f"{prefix}_view_count"] = int(summary.get("num_views", summary.get("n_images", 0)))

        self.metrics_by_iteration[str(int(iteration))] = entry
        doc = {
            "iterations": int(self.total_iterations if self.total_iterations is not None else iteration),
            "source_path_original": self.source_path_original,
            "source_path_after_split": self.source_path_after_split,
            "external_test_source_path": self.external_test_source_path,
            "split_report_path": self.split_report_path,
            "validation_report_path": self.validation_report_path,
            "split_protocol": split_meta.get("protocol"),
            "split_init_policy": split_meta.get("split_init_policy"),
            "split_train_views": split_meta.get("split_train_views"),
            "split_hold": split_meta.get("split_hold"),
            "train_camera_count": self.train_count,
            "test_camera_count": self.test_count,
            "metrics_compute_lpips": self.metrics_compute_lpips,
            "metrics": self.metrics_by_iteration,
        }
        with open(self.run_manifest_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, allow_nan=False)

    def write_iteration(self, iteration, train_scalars, eval_summaries, per_view_rows=None):
        if not eval_summaries:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        num_gaussians = train_scalars.get("num_gaussians", float("nan"))

        lines = []
        for split_name, summary in eval_summaries.items():
            record = self._summary_record(iteration, timestamp, split_name, summary)
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._write_summary_row(record)
            self._write_summary_tsv_row(record)
            self.summary_records.append(record)

            lines.append(f"[ITER {iteration}][{split_name}]")
            lines.append(f"views={record['num_views']}")
            lines.append(f"num_gaussians={record['num_gaussians']}")
            lines.append(f"L1={_format_float(record['l1'], 6)}")
            lines.append(f"MSE={_format_float(record['mse'], 6)}")
            lines.append(f"RMSE={_format_float(record['rmse'], 6)}")
            lines.append(f"PSNR={_format_float(record['psnr'], 6)}")
            lines.append(f"SSIM={_format_float(record['ssim'], 6)}")
            lines.append(f"LPIPS={_format_optional_metric(record['lpips'], record['lpips_status'], 6)}")
            lines.append(f"lpips_status={record['lpips_status']}")
            lines.append(f"split_report={record['split_report']}")
            lines.append("")

        with open(self.readable_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self._write_metrics_summary_json()
        self._update_run_manifest(iteration, eval_summaries)

        if per_view_rows:
            combined_per_view_path = os.path.join(
                self.metrics_dir,
                f"per_view_metrics_{int(iteration)}.csv",
            )
            _maybe_write_header(
                combined_per_view_path,
                ",".join([
                    "iteration",
                    "split",
                    "view_index",
                    "image_name",
                    "num_gaussians",
                    "l1",
                    "mse",
                    "rmse",
                    "psnr",
                    "ssim",
                    "lpips",
                    "lpips_status",
                ])
            )
            for r in per_view_rows:
                lpips_value = r.get("lpips")
                out = [
                    int(iteration),
                    self._quote_csv(r.get("split", "")),
                    int(r.get("view_index", 0)),
                    self._quote_csv(r.get("image_name", "")),
                    int(r.get("num_gaussians", num_gaussians)),
                    self._csv_metric(r.get("l1")),
                    self._csv_metric(r.get("mse")),
                    self._csv_metric(r.get("rmse")),
                    self._csv_metric(r.get("psnr")),
                    self._csv_metric(r.get("ssim")),
                    self._csv_metric(lpips_value),
                    self._quote_csv(r.get("lpips_status", "disabled" if self._json_metric(lpips_value) is None else "enabled")),
                ]
                _append_line(combined_per_view_path, ",".join(map(str, out)))

            rows_by_split = {}
            for row in per_view_rows:
                rows_by_split.setdefault(row.get("split", "unknown"), []).append(row)

            for split_name, rows in rows_by_split.items():
                per_view_path = os.path.join(
                    self.model_path,
                    f"metrics_per_view_iter_{int(iteration):05d}_{split_name}.csv",
                )
                _maybe_write_header(
                    per_view_path,
                    ",".join([
                        "iteration",
                        "split",
                        "view_index",
                        "image_name",
                        "num_gaussians",
                        "l1",
                        "mse",
                        "rmse",
                        "psnr",
                        "ssim",
                        "lpips",
                        "lpips_status",
                    ])
                )
                for r in rows:
                    lpips_value = r.get("lpips")
                    out = [
                        int(iteration),
                        self._quote_csv(split_name),
                        int(r.get("view_index", 0)),
                        self._quote_csv(r.get("image_name", "")),
                        int(r.get("num_gaussians", num_gaussians)),
                        self._csv_metric(r.get("l1")),
                        self._csv_metric(r.get("mse")),
                        self._csv_metric(r.get("rmse")),
                        self._csv_metric(r.get("psnr")),
                        self._csv_metric(r.get("ssim")),
                        self._csv_metric(lpips_value),
                        self._quote_csv(r.get("lpips_status", "disabled" if self._json_metric(lpips_value) is None else "enabled")),
                    ]
                    _append_line(per_view_path, ",".join(map(str, out)))


@torch.no_grad()
def evaluate_camera_set_for_metrics(
    split_name,
    cameras,
    scene,
    renderFunc,
    renderArgs,
    lambda_dssim,
    train_test_exp,
    iteration,
    num_gaussians,
    lpips_model=None,
    lpips_status="disabled",
):
    if cameras is None or len(cameras) == 0:
        return None, []

    sums = {
        "l1": 0.0,
        "mse": 0.0,
        "rmse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "dssim_loss": 0.0,
        "rgb_loss": 0.0,
        "lpips": 0.0,
    }

    per_view_rows = []

    for view_index, viewpoint in enumerate(cameras):
        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

        if train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]

        image_b = image.unsqueeze(0)
        gt_b = gt_image.unsqueeze(0)

        l1_v = torch.abs(image - gt_image).mean()
        mse_v = torch.mean((image - gt_image) ** 2)
        rmse_v = torch.sqrt(torch.clamp(mse_v, min=1e-12))
        psnr_v = -10.0 * torch.log10(torch.clamp(mse_v, min=1e-12))
        ssim_v = ssim(image_b, gt_b)
        dssim_v = 1.0 - ssim_v
        rgb_loss_v = (1.0 - lambda_dssim) * l1_v + lambda_dssim * dssim_v

        image_name = getattr(viewpoint, "image_name", "unknown")
        lpips_v = _compute_lpips_value(lpips_model, image, gt_image)

        row = {
            "split": split_name,
            "view_index": view_index,
            "image_name": image_name,
            "num_gaussians": int(num_gaussians),
            "l1": _finite_or_raise("l1", _safe_metric_value(l1_v), iteration, split_name, image_name),
            "mse": _finite_or_raise("mse", _safe_metric_value(mse_v), iteration, split_name, image_name),
            "rmse": _finite_or_raise("rmse", _safe_metric_value(rmse_v), iteration, split_name, image_name),
            "psnr": _finite_or_raise("psnr", _safe_metric_value(psnr_v), iteration, split_name, image_name),
            "ssim": _finite_or_raise("ssim", _safe_metric_value(ssim_v), iteration, split_name, image_name),
            "dssim_loss": _finite_or_raise("dssim_loss", _safe_metric_value(dssim_v), iteration, split_name, image_name),
            "rgb_loss": _finite_or_raise("rgb_loss", _safe_metric_value(rgb_loss_v), iteration, split_name, image_name),
            "lpips": _finite_or_raise("lpips", lpips_v, iteration, split_name, image_name) if lpips_v is not None else None,
            "lpips_status": lpips_status,
        }
        per_view_rows.append(row)

        sums["l1"] += row["l1"]
        sums["mse"] += row["mse"]
        sums["rmse"] += row["rmse"]
        sums["psnr"] += row["psnr"]
        sums["ssim"] += row["ssim"]
        sums["dssim_loss"] += row["dssim_loss"]
        sums["rgb_loss"] += row["rgb_loss"]

        if row["lpips"] is not None:
            sums["lpips"] += row["lpips"]

    n = len(cameras)
    valid_lpips = [row["lpips"] for row in per_view_rows if row["lpips"] is not None]
    lpips_summary = sum(valid_lpips) / len(valid_lpips) if valid_lpips else None
    summary = {
        "n_images": n,
        "num_views": n,
        "num_gaussians": int(num_gaussians),
        "l1": sums["l1"] / n,
        "mse": sums["mse"] / n,
        "rmse": sums["rmse"] / n,
        "psnr": sums["psnr"] / n,
        "ssim": sums["ssim"] / n,
        "dssim_loss": sums["dssim_loss"] / n,
        "rgb_loss": sums["rgb_loss"] / n,
        "lpips": lpips_summary,
        "lpips_status": lpips_status,
    }

    return summary, per_view_rows

def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    metrics_log_interval=1000,
    metrics_eval_train_count=-1,
    metrics_eval_per_view=False,
    metrics_compute_lpips=False,
    lpips_model=None,
    lpips_status="disabled",
    metrics_eval_test_manifest="",
):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    if getattr(dataset, "split_only", False):
        print("[BASE-SPLIT] split_only=true; exiting before training.")
        return
    pgdr_cache = None
    if getattr(dataset, "pgdr_enable", False):
        pgdr_depth_cache_dir = resolve_depth_cache_dir(
            getattr(dataset, "source_path", ""),
            getattr(dataset, "pgdr_depth_cache", "") or "",
        )
        pgdr_cache = load_pgdr_depth_cache(
            pgdr_depth_cache_dir,
            scene.getTrainCameras(),
            dataset.model_path,
            strict=bool(getattr(dataset, "pgdr_strict_precheck", True)),
            device="cuda",
        )
    lpips_model, lpips_status = _init_lpips_model(
        Namespace(metrics_compute_lpips=metrics_compute_lpips),
        device="cuda",
    )
    metrics_logger = TrainMetricsFileLogger(
        dataset.model_path,
        train_count=len(scene.getTrainCameras()),
        test_count=len(scene.getTestCameras()),
        split_report_path=getattr(scene, "split_report_path", None),
        validation_report_path=getattr(dataset, "auto_split_validation_report_path", None),
        source_path_original=getattr(dataset, "source_path_original", None),
        source_path_after_split=getattr(dataset, "source_path", None),
        external_test_source_path=getattr(dataset, "external_test_source_path", None),
        total_iterations=getattr(opt, "iterations", None),
        metrics_compute_lpips=metrics_compute_lpips,
    )
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = _load_checkpoint_compat(checkpoint)
        gaussians.restore(model_params, opt)
    if getattr(opt, "dropgaussian_enable", False):
        _dropgaussian_rate(opt, max(first_iter, 0))
        print(
            "[DropGaussian] enabled schedule={} start={} end={} max_rate={:.4f}".format(
                opt.dropgaussian_schedule,
                opt.dropgaussian_start,
                opt.dropgaussian_end,
                opt.dropgaussian_max_rate,
            )
        )
    pgdr_state = None
    if getattr(dataset, "pgdr_enable", False) and pgdr_cache is not None and pgdr_cache.get("enabled", False):
        pgdr_config = build_pgdr_config(opt)
        pgdr_state = PGDRState(
            pgdr_cache,
            scene.getTrainCameras(),
            dataset.model_path,
            pgdr_config,
        )
        pgdr_state.resize_or_reset(gaussians.get_xyz.shape[0])
        print(
            "[PGDR] enabled update_interval={} sample_views={} pixels_per_view={} gate_start={}".format(
                pgdr_config.update_interval,
                pgdr_config.sample_views,
                pgdr_config.pixels_per_view,
                pgdr_config.gate_start,
            )
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Load aligned inverse-depth cache (enables the depth-reg loss AND DAD targets). Auto-detected
    # under the train source unless --depth_cache is given. Required for --dad_enable.
    if getattr(opt, "use_depth_cache", False) or getattr(opt, "dad_enable", False) or getattr(opt, "cfdc_enable", False) or getattr(opt, "hog_enable", False):
        _dcache = getattr(opt, "depth_cache", "") or os.path.join(getattr(dataset, "source_path", ""), "pgdr_depth_cache_aligned")
        if os.path.isdir(_dcache):
            from utils.depthguide import load_aligned_depth_cache
            _nm = load_aligned_depth_cache(scene.getTrainCameras(), _dcache)
            print(f"[DEPTH] aligned depth cache loaded: {_nm}/{len(scene.getTrainCameras())} train cams <- {_dcache}")
        else:
            print(f"[DEPTH][WARN] depth cache dir not found: {_dcache} -> depth-reg/DAD will be inactive")
    if getattr(opt, "cfdc_enable", False) or getattr(opt, "swd_enable", False) or getattr(opt, "hog_enable", False):
        _ccache = getattr(opt, "cfdc_cache", "") or os.path.join(_dcache, "cfdc_confidence.pt")
        if os.path.isdir(_ccache) or os.path.isfile(_ccache):
            from utils.depthguide import load_cfdc_confidence_cache
            _cn = load_cfdc_confidence_cache(scene.getTrainCameras(), _ccache)
            print(f"[CFDC] confidence cache loaded: {_cn}/{len(scene.getTrainCameras())} train cams <- {_ccache}")
        else:
            print(f"[CFDC][WARN] confidence cache not found: {_ccache} -> CFDC weights inactive")

    # ---- ECU: learned evidence-conditioned uncertainty head (replaces fixed WGSD mapping) ----
    ecu_net, ecu_optim, ecu_center, ecu_scale = None, None, None, 1.0
    if getattr(opt, "ecu_enable", False):
        from utils.ecu import EvidenceUncertaintyNet
        ecu_net = EvidenceUncertaintyNet(inputs=str(opt.ecu_inputs), hidden=int(opt.ecu_hidden)).cuda()
        ecu_optim = torch.optim.Adam(ecu_net.parameters(), lr=float(opt.ecu_lr))
        with torch.no_grad():
            _xyz0 = gaussians.get_xyz
            ecu_center = _xyz0.mean(0)
            ecu_scale = float((_xyz0 - ecu_center).norm(dim=1).median()) * 3.0
        print(f"[ECU] enabled inputs={opt.ecu_inputs} hidden={opt.ecu_hidden} "
              f"u_target={opt.ecu_u_target} reg={opt.ecu_reg} start={opt.ecu_start} freeze={opt.ecu_freeze_iter}")

    # ---- HOG-GS: held-out-guided regularization policy (outer ES on rotating train folds) ----
    hog_cfg, hog_state = None, None
    if getattr(opt, "hog_enable", False):
        if opt.optimizer_type != "default":
            raise ValueError("[HOG] only optimizer_type=default (plain Adam) is supported (probe inner steps)")
        from utils.hog import HOGConfig, HOGState
        hog_cfg = HOGConfig(opt)
        hog_state = HOGState(hog_cfg, scene.getTrainCameras(), dataset.model_path)
        print(f"[HOG] {hog_cfg} folds={len(hog_state.query_folds)} "
              f"support/query={len(hog_state.support_folds[0])}/{len(hog_state.query_folds[0])}")

    def _hog_render(cam, rates):
        return render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp,
                      separate_sh=SPARSE_ADAM_AVAILABLE,
                      gaussian_dropout_rate=(rates if rates is not None else 0.0))

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # HOG-GS outer step: antithetic ES probe on a rotating held-out fold (runs BEFORE this
        # iteration's real update; snapshots/restores params+Adam state internally).
        if (hog_state is not None and iteration >= hog_cfg.start
                and iteration % hog_cfg.meta_interval == 0):
            if hog_cfg.mode == "grad":
                _hs = hog_state.harvest_and_update(gaussians, _hog_render, iteration)
                if hog_state.meta_count % 10 == 0:
                    print(f"\n[HOG] harvest#{_hs['harvest']} fold={_hs['fold']} n_vis={_hs['n_visible']} "
                          f"frac_harm={_hs['frac_harmful']:.3f} corr_front={_hs.get('corr_front', 'na')} "
                          f"corr_witness={_hs.get('corr_witness', 'na')} mean_rate={_hs['mean_rate']:.3f}")
            else:
                _hs = hog_state.meta_step(gaussians, _hog_render, iteration, float(depth_l1_weight(iteration)))
                if hog_state.meta_count % 10 == 0:
                    print(f"\n[HOG] meta#{_hs['meta_step']} fold={_hs['fold']} dL={_hs['dL']:+.5f} "
                          f"mean_rate={_hs['mean_rate']:.3f} phi={_hs['phi']}")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        # HOG holdout window: in the W iterations before each harvest, the upcoming fold's query
        # views are EXCLUDED from training so the harvested held-out signal is honest (the model has
        # not just been fitted on them). Bounded cost: W/interval of each view's training budget.
        if (hog_state is not None and hog_cfg.mode == "grad" and hog_cfg.holdout_window > 0
                and iteration >= hog_cfg.start - hog_cfg.holdout_window
                and (iteration % hog_cfg.meta_interval) >= hog_cfg.meta_interval - hog_cfg.holdout_window):
            _held = hog_state.upcoming_query_names()
            for _try in range(10):
                if getattr(viewpoint_stack[rand_idx], "image_name", "") not in _held:
                    break
                rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        gaussian_dropout_rate = _dropgaussian_rate(opt, iteration)
        # WGSD: witness-guided structural dropout. Attacks DropGaussian's own stated limitations
        # ("blind randomness can kill important Gaussians"; "no multi-view awareness"): per-Gaussian
        # drop rate modulated by cross-view surface witness - multi-view-confirmed surface Gaussians
        # are dropped LESS (protect verified detail), depth-contradicted (front-floater) Gaussians
        # are dropped MORE. Mean-1 renormalized so the AVERAGE rate stays the scheduled base rate.
        if (getattr(opt, "wgsd_enable", False) and gaussian_dropout_rate > 0.0
                and getattr(gaussians, "_swd_scores", None) is not None
                and gaussians._swd_scores["front_ratio"].shape[0] == gaussians.get_xyz.shape[0]):
            with torch.no_grad():
                sc = gaussians._swd_scores
                signal = (sc["front_ratio"] - sc["witness"]).clamp(-1.0, 1.0)   # + = suspect, - = verified
                mod = torch.exp(float(opt.wgsd_beta) * signal)
                mod = torch.where(sc["evidence"], mod, torch.ones_like(mod))    # no evidence -> neutral
                mod = mod / mod.mean().clamp(min=1e-6)                          # mean-1
                gaussian_dropout_rate = (gaussian_dropout_rate * mod).clamp(0.0, float(opt.wgsd_max_rate))
        # HOG-GS: per-Gaussian rates from the held-out-learned policy (tensor path of the renderer's
        # inverted dropout keeps E[opacity] unbiased; replaces uniform/WGSD when enabled)
        if hog_state is not None and iteration >= hog_cfg.start:
            gaussian_dropout_rate = (hog_state.applied_rates(gaussians, iteration) if hog_cfg.mode == "grad"
                                     else hog_state.applied_rates_es(gaussians, iteration))
        # ECU: learned per-Gaussian uncertainty -> differentiable train-time opacity multiplier
        ecu_multiplier = None
        ecu_u = None
        if (ecu_net is not None and iteration >= int(opt.ecu_start)):
            from utils.ecu import ecu_train_multiplier
            _swd_sc = getattr(gaussians, "_swd_scores", None)
            _swd_ok = (_swd_sc is not None and _swd_sc["front_ratio"].shape[0] == gaussians.get_xyz.shape[0])
            # evidence/both need size-matched SWD scores; shape-only never does. Skip (multiplier=None)
            # in the iterations right after a densify until SWD refreshes - never crash, never go stale.
            if ecu_net.inputs == "shape" or _swd_ok:
                feats = ecu_net.build_features(gaussians, _swd_sc if _swd_ok else None, ecu_center, ecu_scale)
                if iteration >= int(opt.ecu_freeze_iter):
                    with torch.no_grad():
                        ecu_u = ecu_net(feats)
                else:
                    ecu_u = ecu_net(feats)
                ecu_multiplier = ecu_train_multiplier(ecu_u, tau=float(opt.ecu_tau))
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            gaussian_dropout_rate=gaussian_dropout_rate,
            opacity_multiplier=ecu_multiplier,
        )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        dssim_loss_value = 1.0 - ssim_value
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * dssim_loss_value
        loss = rgb_loss

        pgdr_update_summary = {}
        pgdr_loss_scalars = {
            "pgdr_pull_count": 0,
            "pgdr_death_candidate_count": 0,
            "pgdr_certified_count": 0,
            "pgdr_birth_hold_count": 0,
            "pgdr_pull_loss": 0.0,
            "pgdr_death_loss": 0.0,
        }
        pgdr_pull_loss = torch.tensor(0.0, device="cuda")
        pgdr_death_loss = torch.tensor(0.0, device="cuda")
        swd_opacity_loss = torch.tensor(0.0, device="cuda")
        if pgdr_state is not None:
            if pgdr_state.should_update(iteration):
                pgdr_update_summary = pgdr_state.update(gaussians, iteration)
            pgdr_pull_loss, pgdr_death_loss, pgdr_loss_scalars = pgdr_state.compute_losses(gaussians, iteration)
            loss += pgdr_pull_loss + pgdr_death_loss

        if getattr(opt, "swd_enable", False) and iteration >= int(opt.swd_start):
            should_refresh_swd = (
                not hasattr(gaussians, "_swd_scores")
                or gaussians._swd_scores is None
                or gaussians._swd_scores["front_ratio"].shape[0] != gaussians.get_xyz.shape[0]
                or iteration % int(opt.swd_update_interval) == 0
            )
            if should_refresh_swd:
                from utils.depthguide import compute_surface_witness_scores
                gaussians._swd_scores = compute_surface_witness_scores(
                    gaussians,
                    scene.getTrainCameras(),
                    tau=float(opt.swd_tau),
                    conf_threshold=float(opt.swd_conf_threshold),
                    min_seen=int(opt.swd_min_seen),
                    front_ratio_threshold=float(opt.swd_front_ratio),
                )
            swd_scores = gaussians._swd_scores
            swd_mask = swd_scores["contradiction"].detach()
            if bool(swd_mask.any()):
                swd_weight = swd_scores["front_ratio"].detach().clamp(0.0, 1.0)
                op = gaussians.get_opacity.squeeze()
                swd_opacity_loss = float(opt.swd_lambda_opacity) * ((op[swd_mask] ** 2) * swd_weight[swd_mask]).mean()
                loss += swd_opacity_loss

        # Depth regularization
        depth_weight_now = depth_l1_weight(iteration)
        Ll1depth_pure = torch.tensor(0.0, device="cuda")
        Ll1depth_weighted = torch.tensor(0.0, device="cuda")

        if depth_weight_now > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            # CDW: complementary depth weighting. Redistribute depth supervision toward TEXTURELESS
            # regions (photometric weak -> depth should lead, where the structural error lives) and
            # away from TEXTURED regions (photometric reliable -> defer to it). Mean-1 renormalized so
            # only the SPATIAL emphasis changes, not the overall depth strength. Soft -> cannot destroy.
            w_cdw = 1.0
            if getattr(opt, "cdw_enable", False):
                with torch.no_grad():
                    gt_g = viewpoint_cam.original_image.cuda().mean(0, keepdim=True)   # [1,H,W]
                    gx = torch.zeros_like(gt_g); gy = torch.zeros_like(gt_g)
                    gx[:, :, 1:] = (gt_g[:, :, 1:] - gt_g[:, :, :-1]).abs()
                    gy[:, 1:, :] = (gt_g[:, 1:, :] - gt_g[:, :-1, :]).abs()
                    tex = (gx + gy)
                    tex = tex / tex.mean().clamp(min=1e-6)
                    w_cdw = torch.exp(-float(opt.cdw_gamma) * tex)
                    w_cdw = w_cdw / w_cdw.mean().clamp(min=1e-6)                        # mean-1
            if getattr(opt, "cfdc_enable", False) and getattr(viewpoint_cam, "depth_confidence", None) is not None:
                with torch.no_grad():
                    conf = viewpoint_cam.depth_confidence.cuda().float().clamp(0.0, 1.0)
                    if conf.dim() == 2:
                        conf = conf[None]
                    w_cfdc = float(opt.cfdc_floor) + (1.0 - float(opt.cfdc_floor)) * conf.pow(float(opt.cfdc_power))
                    valid = depth_mask.float()
                    denom = valid.sum().clamp(min=1.0)
                    mean_valid = (w_cfdc * valid).sum() / denom
                    w_cfdc = w_cfdc / mean_valid.clamp(min=1e-6)
                    w_cdw = w_cdw * w_cfdc

            Ll1depth_pure = (torch.abs((invDepth  - mono_invdepth) * depth_mask) * w_cdw).mean()
            Ll1depth_weighted = depth_weight_now * Ll1depth_pure
            loss += Ll1depth_weighted

        Ll1depth = _safe_metric_value(Ll1depth_weighted)

        # ECU rate anchor: keep MEAN suppression at the DropGaussian-equivalent level so the net
        # learns the ALLOCATION, not "no dropout" (render loss alone prefers u->0 collapse).
        if ecu_u is not None and ecu_u.requires_grad:
            loss = loss + float(opt.ecu_reg) * (ecu_u.mean() - float(opt.ecu_u_target)) ** 2

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            loss_item = loss.item()
            elapsed_ms = _cuda_elapsed_ms(iter_start, iter_end)
            train_scalars = {
                "train_l1": _safe_metric_value(Ll1),
                "train_ssim": _safe_metric_value(ssim_value),
                "train_dssim_loss": _safe_metric_value(dssim_loss_value),
                "train_rgb_loss": _safe_metric_value(rgb_loss),
                "train_depth_l1_pure": _safe_metric_value(Ll1depth_pure),
                "train_depth_weight": float(depth_weight_now),
                "train_depth_loss": _safe_metric_value(Ll1depth_weighted),
                "dropgaussian_rate": float(gaussian_dropout_rate.mean()) if torch.is_tensor(gaussian_dropout_rate) else float(gaussian_dropout_rate),
                "pgdr_pull_loss": _safe_metric_value(pgdr_pull_loss),
                "pgdr_death_loss": _safe_metric_value(pgdr_death_loss),
                "swd_opacity_loss": _safe_metric_value(swd_opacity_loss),
                "pgdr_last_update_iteration": int(pgdr_state.last_update_iteration) if pgdr_state is not None else 0,
                "pgdr_last_pixels_used": int(pgdr_update_summary.get("pixels_used", 0)) if pgdr_update_summary else 0,
                "train_total_loss": _safe_metric_value(loss),
                "elapsed_ms": _safe_metric_value(elapsed_ms),
                "num_gaussians": int(gaussians.get_xyz.shape[0]),
            }
            train_scalars.update(pgdr_loss_scalars)

            # Progress bar
            ema_loss_for_log = 0.4 * loss_item + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                elapsed_ms,
                testing_iterations,
                scene,
                render,
                (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp,
                train_scalars=train_scalars,
                metrics_logger=metrics_logger,
                lambda_dssim=opt.lambda_dssim,
                metrics_log_interval=metrics_log_interval,
                metrics_eval_train_count=metrics_eval_train_count,
                metrics_eval_per_view=metrics_eval_per_view,
                metrics_compute_lpips=metrics_compute_lpips,
                lpips_model=lpips_model,
                lpips_status=lpips_status,
                metrics_eval_test_manifest=metrics_eval_test_manifest,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    pgdr_certified_mask = pgdr_state.adc_certified_mask(iteration) if pgdr_state is not None else None
                    if getattr(opt, "swd_enable", False) and getattr(opt, "swd_birth_gate", False) and hasattr(gaussians, "_swd_scores") and gaussians._swd_scores is not None:
                        swd_allow_mask = torch.logical_not(gaussians._swd_scores["contradiction"]).detach()
                        if pgdr_certified_mask is None:
                            pgdr_certified_mask = swd_allow_mask
                        else:
                            nmask = min(pgdr_certified_mask.shape[0], swd_allow_mask.shape[0])
                            merged = pgdr_certified_mask.clone()
                            merged[:nmask] = torch.logical_and(merged[:nmask].bool(), swd_allow_mask[:nmask].bool())
                            pgdr_certified_mask = merged
                    # DAD: refresh per-Gaussian depth-prior targets (multi-view consensus) before splitting
                    if getattr(opt, "dad_enable", False):
                        from utils.depthguide import compute_depth_targets
                        gaussians.dad_enabled = True
                        gaussians.dad_alpha = float(opt.dad_alpha)
                        gaussians._dad_target, gaussians._dad_reliable = compute_depth_targets(
                            gaussians, scene.getTrainCameras(),
                            min_views=int(opt.dad_min_views), agree_rel=float(opt.dad_agree))
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        pgdr_certified_mask=pgdr_certified_mask,
                    )
                    if pgdr_state is not None:
                        pgdr_state.resize_or_reset(gaussians.get_xyz.shape[0])
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if ecu_optim is not None and ecu_u is not None and ecu_u.requires_grad:
                    ecu_optim.step()
                    ecu_optim.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp,
    train_scalars=None,
    metrics_logger=None,
    lambda_dssim=0.2,
    metrics_log_interval=1000,
    metrics_eval_train_count=-1,
    metrics_eval_per_view=False,
    metrics_compute_lpips=False,
    lpips_model=None,
    lpips_status="disabled",
    metrics_eval_test_manifest="",
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if tb_writer and train_scalars is not None:
        tb_writer.add_scalar("train_loss_patches/ssim", train_scalars["train_ssim"], iteration)
        tb_writer.add_scalar("train_loss_patches/dssim_loss", train_scalars["train_dssim_loss"], iteration)
        tb_writer.add_scalar("train_loss_patches/rgb_loss", train_scalars["train_rgb_loss"], iteration)
        tb_writer.add_scalar("train_loss_patches/depth_l1_pure", train_scalars["train_depth_l1_pure"], iteration)
        tb_writer.add_scalar("train_loss_patches/depth_weight", train_scalars["train_depth_weight"], iteration)
        tb_writer.add_scalar("train_loss_patches/depth_loss", train_scalars["train_depth_loss"], iteration)
        tb_writer.add_scalar("train_loss_patches/swd_opacity_loss", train_scalars.get("swd_opacity_loss", 0.0), iteration)
        tb_writer.add_scalar("pgdr/pull_loss", train_scalars.get("pgdr_pull_loss", 0.0), iteration)
        tb_writer.add_scalar("pgdr/death_loss", train_scalars.get("pgdr_death_loss", 0.0), iteration)
        tb_writer.add_scalar("pgdr/certified_count", train_scalars.get("pgdr_certified_count", 0), iteration)
        tb_writer.add_scalar("pgdr/birth_hold_count", train_scalars.get("pgdr_birth_hold_count", 0), iteration)
        tb_writer.add_scalar("pgdr/pull_count", train_scalars.get("pgdr_pull_count", 0), iteration)
        tb_writer.add_scalar("pgdr/death_candidate_count", train_scalars.get("pgdr_death_candidate_count", 0), iteration)
        tb_writer.add_scalar("scene/total_points", train_scalars["num_gaussians"], iteration)

    should_eval = iteration in testing_iterations or (metrics_log_interval > 0 and (iteration % metrics_log_interval == 0))
    should_train_log = False

    if should_eval:
        torch.cuda.empty_cache()
        first_testing_iteration = min((test_iteration for test_iteration in testing_iterations if test_iteration > 0), default=None)

        train_cameras = sorted(scene.getTrainCameras(), key=lambda c: getattr(c, "image_name", ""))
        test_cameras = sorted(scene.getTestCameras(), key=lambda c: getattr(c, "image_name", ""))
        test_split_name = "test"
        if metrics_eval_test_manifest:
            test_cameras = _select_manifest_cameras(test_cameras, metrics_eval_test_manifest)
            test_split_name = "validation"
        num_gaussians = int(scene.gaussians.get_xyz.shape[0])

        if metrics_eval_train_count < 0:
            train_eval_cameras = train_cameras
            train_split_name = "train"
        elif metrics_eval_train_count == 0 or len(train_cameras) == 0:
            train_eval_cameras = []
            train_split_name = "train"
        else:
            train_eval_cameras = train_cameras[:metrics_eval_train_count]
            train_split_name = "train"

        validation_configs = [
            {"name": test_split_name, "cameras": test_cameras},
            {"name": train_split_name, "cameras": train_eval_cameras},
        ]

        eval_summaries = {}
        all_per_view_rows = []

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                summary, per_view_rows = evaluate_camera_set_for_metrics(
                    split_name=config["name"],
                    cameras=config["cameras"],
                    scene=scene,
                    renderFunc=renderFunc,
                    renderArgs=renderArgs,
                    lambda_dssim=lambda_dssim,
                    train_test_exp=train_test_exp,
                    iteration=iteration,
                    num_gaussians=num_gaussians,
                    lpips_model=lpips_model,
                    lpips_status=lpips_status,
                )

                if summary is not None:
                    eval_summaries[config["name"]] = summary

                    if metrics_eval_per_view:
                        all_per_view_rows.extend(per_view_rows)

                    print(
                        "\n[ITER {}] Evaluating {}: "
                        "L1 {:.6f} MSE {:.6f} RMSE {:.6f} PSNR {:.6f} SSIM {:.6f} LPIPS {} NumGaussians {}".format(
                            iteration,
                            config["name"],
                            summary["l1"],
                            summary["mse"],
                            summary["rmse"],
                            summary["psnr"],
                            summary["ssim"],
                            _format_optional_metric(summary.get("lpips"), summary.get("lpips_status", lpips_status), 6),
                            summary["num_gaussians"],
                        )
                    )

                    if tb_writer:
                        for idx, viewpoint in enumerate(config["cameras"][:5]):
                            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            if train_test_exp:
                                image = image[..., image.shape[-1] // 2:]
                                gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                            tb_writer.add_images(config["name"] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if first_testing_iteration is not None and iteration == first_testing_iteration:
                                tb_writer.add_images(config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        tb_writer.add_scalar(config["name"] + "/l1", summary["l1"], iteration)
                        tb_writer.add_scalar(config["name"] + "/mse", summary["mse"], iteration)
                        tb_writer.add_scalar(config["name"] + "/rmse", summary["rmse"], iteration)
                        tb_writer.add_scalar(config["name"] + "/psnr", summary["psnr"], iteration)
                        tb_writer.add_scalar(config["name"] + "/ssim", summary["ssim"], iteration)
                        tb_writer.add_scalar(config["name"] + "/dssim_loss", summary["dssim_loss"], iteration)
                        tb_writer.add_scalar(config["name"] + "/rgb_loss", summary["rgb_loss"], iteration)
                        if summary.get("lpips") is not None:
                            tb_writer.add_scalar(config["name"] + "/lpips_vgg", summary["lpips"], iteration)
                            tb_writer.add_scalar(config["name"] + "/metrics/lpips", summary["lpips"], iteration)

                        tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", summary["l1"], iteration)
                        tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", summary["psnr"], iteration)
                        tb_writer.add_scalar(config["name"] + "/metrics/psnr", summary["psnr"], iteration)
                        tb_writer.add_scalar(config["name"] + "/metrics/ssim", summary["ssim"], iteration)

        if metrics_logger is not None and train_scalars is not None:
            metrics_logger.write_iteration(
                iteration=iteration,
                train_scalars=train_scalars,
                eval_summaries=eval_summaries,
                per_view_rows=all_per_view_rows if metrics_eval_per_view else None,
            )

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()

    elif should_train_log and metrics_logger is not None and train_scalars is not None:
        metrics_logger.write_iteration(
            iteration=iteration,
            train_scalars=train_scalars,
            eval_summaries={},
            per_view_rows=None,
        )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
        "--skip_final_save",
        action="store_true",
        default=False,
        help="Do not automatically save a PLY at --iterations. Useful for short screening runs.",
    )
    parser.add_argument(
        "--metrics_log_interval",
        type=int,
        default=1000,
        help="Evaluate and append metrics every N iterations. Set 0 to use only --test_iterations."
    )
    parser.add_argument(
        "--metrics_eval_train_count",
        type=int,
        default=-1,
        help="Number of sorted train cameras to evaluate. Use -1 for all train cameras, 0 to disable train eval."
    )
    parser.add_argument(
        "--metrics_eval_per_view",
        action="store_true",
        default=False,
        help="Write per-view metrics CSV files at metric iterations."
    )
    parser.add_argument(
        "--metrics_eval_test_manifest",
        type=str,
        default="",
        help=(
            "Optional image-name manifest selecting a strict subset of test cameras for tuning. "
            "Selected metrics are logged under the validation split."
        ),
    )
    parser.add_argument(
        "--metrics_compute_lpips",
        action="store_true",
        default=False,
        help="Compute LPIPS during metrics evaluation. Requires lpips package. If enabled and lpips is unavailable, training fails early."
    )
    parser.add_argument(
        "--metrics_disable_lpips",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag. LPIPS is disabled unless --metrics_compute_lpips is set."
    )
    parser.add_argument(
        "--sparse12_10k",
        action="store_true",
        default=False,
        help="Preset: strict train-only COLMAP split with 12 train views, hold=8 test, 10k iterations, metrics every 1k."
    )
    args = parser.parse_args(sys.argv[1:])
    _apply_sparse12_10k_preset(args)
    if args.metrics_compute_lpips and args.metrics_disable_lpips:
        raise ValueError("[METRICS][LPIPS] Do not combine --metrics_compute_lpips with --metrics_disable_lpips.")
    if args.split_train_views != "off":
        from utils.auto_split_3dgs import prepare_auto_split

        args.source_path_original = args.source_path
        split_result = prepare_auto_split(
            source_path=args.source_path,
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
            split_require_all_train_registered=args.split_require_all_train_registered,
            split_min_train_points=args.split_min_train_points,
            split_min_triangulated_points=args.split_min_triangulated_points,
            split_strict_sparsegs=args.split_strict_sparsegs,
            strict_no_overlap=args.split_strict_no_overlap,
        )

        if split_result["status"] != "PASS":
            raise RuntimeError(
                "Auto split failed. Abort before training. "
                f"reason={split_result.get('reason', split_result.get('failure_reasons', 'unknown'))}"
            )

        args.source_path = split_result["train_source_path"]
        args.external_test_source_path = split_result["test_source_path"]
        args.eval = True
        args.auto_split_report_path = split_result["split_report_path"]
        args.auto_split_validation_report_path = split_result["validation_report_path"]

        print("[AUTO-SPLIT] status=PASS")
        print(f"[AUTO-SPLIT] protocol={split_result.get('protocol')}")
        print(f"[AUTO-SPLIT] train_source={args.source_path}")
        print(f"[AUTO-SPLIT] test_source={args.external_test_source_path}")
        print(f"[AUTO-SPLIT] report={args.auto_split_report_path}")

        if args.split_validate_only:
            print("[AUTO-SPLIT] split_validate_only=True. Exit before training.")
            sys.exit(0)

    if getattr(args, "use_existing_split", False):
        if args.split_train_views != "off":
            raise ValueError("--use_existing_split cannot be combined with --split_train_views.")

        split_root = os.path.abspath(args.source_path)
        train_candidate = os.path.join(split_root, "train")
        test_candidate = os.path.join(split_root, "test")
        report_candidate = os.path.join(split_root, "reports", "split_report.json")
        validation_candidate = os.path.join(split_root, "reports", "validation_report.json")

        if os.path.isdir(train_candidate) and os.path.isdir(test_candidate):
            args.source_path_original = split_root
            args.source_path = train_candidate
            args.external_test_source_path = test_candidate
            if os.path.isfile(report_candidate):
                args.auto_split_report_path = report_candidate
            if os.path.isfile(validation_candidate):
                args.auto_split_validation_report_path = validation_candidate
            args.eval = True
            print("[EXISTING-SPLIT] using split_root={}".format(split_root))
            print("[EXISTING-SPLIT] train_source={}".format(args.source_path))
            print("[EXISTING-SPLIT] test_source={}".format(args.external_test_source_path))
        elif args.external_test_source_path:
            args.eval = True
            print("[EXISTING-SPLIT] using explicit source_path/external_test_source_path")
        else:
            raise ValueError(
                "--use_existing_split expects --source_path to be a split root containing train/ and test/, "
                "or an explicit --external_test_source_path."
            )

    if not args.skip_final_save:
        args.save_iterations.append(args.iterations)
    args.save_iterations = sorted(set(args.save_iterations))
    args.checkpoint_iterations = sorted(set(args.checkpoint_iterations))
    metric_iterations = set(args.test_iterations)
    if args.metrics_log_interval and args.metrics_log_interval > 0:
        metric_iterations.update(range(args.metrics_log_interval, args.iterations + 1, args.metrics_log_interval))
    args.test_iterations = sorted(metric_iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        metrics_log_interval=args.metrics_log_interval,
        metrics_eval_train_count=args.metrics_eval_train_count,
        metrics_eval_per_view=args.metrics_eval_per_view,
        metrics_compute_lpips=args.metrics_compute_lpips,
        metrics_eval_test_manifest=args.metrics_eval_test_manifest,
    )

    # All done
    print("\nTraining complete.")
