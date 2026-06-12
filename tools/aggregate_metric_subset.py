#!/usr/bin/env python3
"""Aggregate per-view 3DGS metrics for an immutable image-name manifest."""

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = ("l1", "mse", "rmse", "psnr", "ssim", "lpips")


def read_manifest(path):
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Manifest contains duplicate image names: {}".format(path))
    return names


def finite_float(value, field, image_name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} is not finite for {}".format(field, image_name))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    names = read_manifest(args.manifest)
    wanted = set(names)
    rows = {}
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != args.split:
                continue
            if args.iteration is not None and int(row["iteration"]) != args.iteration:
                continue
            image_name = row["image_name"]
            if image_name in wanted:
                if image_name in rows:
                    raise ValueError("Duplicate metric row for {}".format(image_name))
                rows[image_name] = row

    missing = [name for name in names if name not in rows]
    if missing:
        raise ValueError("Missing metric rows: {}".format(", ".join(missing)))

    ordered_rows = [rows[name] for name in names]
    aggregate = {
        metric: sum(finite_float(row[metric], metric, row["image_name"]) for row in ordered_rows)
        / len(ordered_rows)
        for metric in METRICS
    }
    result = {
        "csv": str(args.csv.resolve()),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "iteration": args.iteration,
        "view_count": len(ordered_rows),
        "metrics": aggregate,
        "images": names,
    }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
