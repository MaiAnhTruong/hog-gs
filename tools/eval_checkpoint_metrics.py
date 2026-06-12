"""Evaluate a saved training checkpoint without resuming training.

This tool reuses train.py's metric implementation so the output is comparable
to metrics_summary.tsv/run_manifest.json produced during training.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from train import (
    SPARSE_ADAM_AVAILABLE,
    TrainMetricsFileLogger,
    _init_lpips_model,
    _load_checkpoint_compat,
    evaluate_camera_set_for_metrics,
)


def _base_parser():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    return parser, lp, op, pp


def main():
    parser, lp, op, pp = _base_parser()
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_out", required=True)
    parser.add_argument("--iteration", type=int, default=8000)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--per_view", action="store_true")
    args = parser.parse_args()

    split_root = os.path.abspath(args.split_root)
    train_source = os.path.join(split_root, "train")
    test_source = os.path.join(split_root, "test")
    os.makedirs(args.eval_out, exist_ok=True)

    # Populate ModelParams-compatible fields.
    args.source_path = train_source
    args.model_path = args.eval_out
    args.external_test_source_path = test_source
    args.images = "images"
    args.depths = ""
    args.eval = True
    args.train_test_exp = False
    if not hasattr(args, "resolution") or args.resolution is None:
        args.resolution = -1
    if not hasattr(args, "data_device") or args.data_device is None:
        args.data_device = "cuda"

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    model_params, ckpt_iter = _load_checkpoint_compat(args.checkpoint)
    gaussians.restore(model_params, opt)

    lpips_model, lpips_status = _init_lpips_model(
        argparse.Namespace(metrics_compute_lpips=bool(args.compute_lpips)),
        device="cuda",
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_args = (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)
    num_gaussians = int(gaussians.get_xyz.shape[0])

    train_cams = sorted(scene.getTrainCameras(), key=lambda c: getattr(c, "image_name", ""))
    test_cams = sorted(scene.getTestCameras(), key=lambda c: getattr(c, "image_name", ""))

    eval_summaries = {}
    per_view_rows = []
    for split_name, cams in (("test", test_cams), ("train", train_cams)):
        summary, rows = evaluate_camera_set_for_metrics(
            split_name=split_name,
            cameras=cams,
            scene=scene,
            renderFunc=render,
            renderArgs=render_args,
            lambda_dssim=float(opt.lambda_dssim),
            train_test_exp=dataset.train_test_exp,
            iteration=int(args.iteration),
            num_gaussians=num_gaussians,
            lpips_model=lpips_model,
            lpips_status=lpips_status,
        )
        eval_summaries[split_name] = summary
        if args.per_view:
            per_view_rows.extend(rows)
        print(
            "[EVAL][{}] views={} PSNR={:.6f} SSIM={:.6f} LPIPS={} num_gaussians={}".format(
                split_name,
                summary["num_views"],
                summary["psnr"],
                summary["ssim"],
                "NA" if summary.get("lpips") is None else f"{summary['lpips']:.6f}",
                summary["num_gaussians"],
            )
        )

    logger = TrainMetricsFileLogger(
        args.eval_out,
        train_count=len(train_cams),
        test_count=len(test_cams),
        split_report_path=os.path.join(split_root, "reports", "split_report.json"),
        source_path_original=split_root,
        source_path_after_split=train_source,
        external_test_source_path=test_source,
        total_iterations=int(args.iteration),
        metrics_compute_lpips=bool(args.compute_lpips),
    )
    logger.write_iteration(
        iteration=int(args.iteration),
        train_scalars={"num_gaussians": num_gaussians},
        eval_summaries=eval_summaries,
        per_view_rows=per_view_rows if args.per_view else None,
    )

    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_iteration": int(ckpt_iter),
        "reported_iteration": int(args.iteration),
        "num_gaussians": num_gaussians,
        "metrics": eval_summaries,
    }
    with open(os.path.join(args.eval_out, "eval_checkpoint_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
