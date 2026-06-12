import argparse
import csv
import json
import os
from collections import defaultdict

import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.pose_support_sh import (
    angular_gap_degrees,
    camera_direction,
    compose_fixed_degree_color,
    compose_support_gated_color,
    leave_one_out_angular_support,
    sh_degree_components,
    support_quantiles,
)


def read_manifest(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def select_cameras(cameras, names):
    by_name = {camera.image_name: camera for camera in cameras}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError("manifest cameras not found: {}".format(", ".join(missing)))
    return [by_name[name] for name in names]


def parse_float_list(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def build_configs(temperatures, floors, powers, active_degree):
    configs = [{"name": "baseline", "kind": "baseline"}]
    for degree in range(active_degree + 1):
        configs.append(
            {
                "name": "fixed_sh{}".format(degree),
                "kind": "fixed",
                "degree": degree,
            }
        )
    for temperature in temperatures:
        for floor in floors:
            for power in powers:
                configs.append(
                    {
                        "name": "support_t{:g}_f{:g}_p{:g}".format(
                            temperature,
                            floor,
                            power,
                        ),
                        "kind": "support",
                        "temperature": temperature,
                        "floor": floor,
                        "power": power,
                    }
                )
    return configs


def mean(values):
    return float(sum(values) / len(values)) if values else float("nan")


def evaluate_metrics(prediction, target, lpips_model):
    prediction_b = prediction.unsqueeze(0)
    target_b = target.unsqueeze(0)
    result = {
        "psnr": float(psnr(prediction_b, target_b).mean().item()),
        "ssim": float(ssim(prediction_b, target_b).item()),
    }
    if lpips_model is not None:
        pred_lpips = torch.clamp(prediction_b, 0.0, 1.0) * 2.0 - 1.0
        target_lpips = torch.clamp(target_b, 0.0, 1.0) * 2.0 - 1.0
        result["lpips"] = float(lpips_model(pred_lpips, target_lpips).mean().item())
    return result


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Probe pose-support-gated SH on a held-out validation manifest."
    )
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temperatures", default="0.01,0.03,0.1")
    parser.add_argument("--floors", default="0,0.25,0.5")
    parser.add_argument("--powers", default="0.5,1")
    parser.add_argument("--compute_lpips", action="store_true")
    args = parser.parse_args()

    split_root = os.path.abspath(args.source_path)
    train_candidate = os.path.join(split_root, "train")
    test_candidate = os.path.join(split_root, "test")
    if os.path.isdir(train_candidate) and os.path.isdir(test_candidate):
        args.source_path = train_candidate
        args.external_test_source_path = test_candidate
        args.eval = True

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
    )
    train_cameras = scene.getTrainCameras()
    validation_cameras = select_cameras(
        scene.getTestCameras(),
        read_manifest(args.manifest),
    )
    if gaussians.active_sh_degree > 3:
        raise ValueError("probe currently supports SH degree at most 3")

    lpips_model = None
    if args.compute_lpips:
        from lpipsPyTorch.modules.lpips import LPIPS

        lpips_model = LPIPS("vgg", "0.1").cuda().eval()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)

    temperatures = parse_float_list(args.temperatures)
    configs = build_configs(
        temperatures,
        parse_float_list(args.floors),
        parse_float_list(args.powers),
        gaussians.active_sh_degree,
    )
    accumulators = {
        config["name"]: defaultdict(list)
        for config in configs
    }
    per_view_rows = []
    support_rows = []

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    xyz = gaussians.get_xyz
    features = gaussians.get_features.transpose(1, 2).contiguous()

    with torch.no_grad():
        for camera in validation_cameras:
            directions = camera_direction(xyz, camera.camera_center)
            components = sh_degree_components(
                features,
                directions,
                gaussians.active_sh_degree,
            )
            support_cache = {}
            for temperature in temperatures:
                support, max_cosine = leave_one_out_angular_support(
                    xyz,
                    camera,
                    train_cameras,
                    temperature,
                )
                support_cache[temperature] = support
                gaps = angular_gap_degrees(max_cosine)
                row = {
                    "image": camera.image_name,
                    "temperature": temperature,
                    "gap_mean_deg": float(gaps.mean().item()),
                    "gap_median_deg": float(gaps.median().item()),
                    "gap_q90_deg": float(torch.quantile(gaps, 0.9).item()),
                }
                row.update(support_quantiles(support))
                support_rows.append(row)

            target = camera.original_image.cuda()
            for config in configs:
                if config["kind"] == "baseline":
                    prediction = render(
                        camera,
                        gaussians,
                        pipe,
                        background,
                    )["render"]
                elif config["kind"] == "fixed":
                    colors = compose_fixed_degree_color(
                        components,
                        config["degree"],
                    )
                    prediction = render(
                        camera,
                        gaussians,
                        pipe,
                        background,
                        override_color=colors,
                    )["render"]
                else:
                    colors = compose_support_gated_color(
                        components,
                        support_cache[config["temperature"]],
                        config["floor"],
                        config["power"],
                    )
                    prediction = render(
                        camera,
                        gaussians,
                        pipe,
                        background,
                        override_color=colors,
                    )["render"]

                metrics = evaluate_metrics(prediction, target, lpips_model)
                for metric_name, metric_value in metrics.items():
                    accumulators[config["name"]][metric_name].append(metric_value)
                per_view_row = {
                    "config": config["name"],
                    "image": camera.image_name,
                }
                per_view_row.update(metrics)
                per_view_rows.append(per_view_row)

            del components
            del support_cache
            torch.cuda.empty_cache()

    summary_rows = []
    baseline_summary = {}
    for config in configs:
        values = accumulators[config["name"]]
        row = {
            "config": config["name"],
            "psnr": mean(values["psnr"]),
            "ssim": mean(values["ssim"]),
        }
        if args.compute_lpips:
            row["lpips"] = mean(values["lpips"])
        if config["name"] == "baseline":
            baseline_summary = dict(row)
        summary_rows.append(row)

    for row in summary_rows:
        row["delta_psnr"] = row["psnr"] - baseline_summary["psnr"]
        row["delta_ssim"] = row["ssim"] - baseline_summary["ssim"]
        if args.compute_lpips:
            row["delta_lpips"] = row["lpips"] - baseline_summary["lpips"]

    summary_rows.sort(key=lambda row: row["psnr"], reverse=True)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_fields = [
        "config",
        "psnr",
        "ssim",
    ]
    per_view_fields = ["config", "image", "psnr", "ssim"]
    if args.compute_lpips:
        summary_fields.append("lpips")
        per_view_fields.append("lpips")
    summary_fields.extend(["delta_psnr", "delta_ssim"])
    if args.compute_lpips:
        summary_fields.append("delta_lpips")

    write_csv(
        os.path.join(args.output_dir, "summary.csv"),
        summary_rows,
        summary_fields,
    )
    write_csv(
        os.path.join(args.output_dir, "per_view.csv"),
        per_view_rows,
        per_view_fields,
    )
    write_csv(
        os.path.join(args.output_dir, "support_stats.csv"),
        support_rows,
        list(support_rows[0].keys()),
    )
    with open(
        os.path.join(args.output_dir, "config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "source_path": split_root,
                "model_path": dataset.model_path,
                "iteration": args.iteration,
                "manifest": os.path.abspath(args.manifest),
                "validation_views": [camera.image_name for camera in validation_cameras],
                "train_views": [camera.image_name for camera in train_cameras],
                "configs": configs,
            },
            handle,
            indent=2,
        )

    for row in summary_rows[:10]:
        print(
            "{config}: PSNR={psnr:.6f} ({delta_psnr:+.6f}) "
            "SSIM={ssim:.6f} ({delta_ssim:+.6f})".format(**row)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
