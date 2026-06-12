"""Depth-Anything-V2 inference for a split's train images.

Writes per-image raw relative-inverse-depth .npy (float32, native image resolution) +
depth_anything_v2_manifest.json into <out_dir>. These raws are the input of
tools/build_depth_cache.py (which aligns them to COLMAP scale).

Requires a Depth-Anything-V2 checkout with checkpoints (default Lightning layout:
/teamspace/studios/this_studio/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth).

Usage:
  python tools/run_da2_depth.py --img_dir <split>/train/images \
      --out_dir <split>/train/depth_anything_v2_vitl \
      --da2_dir /teamspace/studios/this_studio/Depth-Anything-V2
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--da2_dir", default="/teamspace/studios/this_studio/Depth-Anything-V2")
    ap.add_argument("--encoder", default="vitl", choices=list(MODEL_CONFIGS))
    ap.add_argument("--checkpoint", default="", help="default: <da2_dir>/checkpoints/depth_anything_v2_<encoder>.pth")
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = args.checkpoint or os.path.join(args.da2_dir, "checkpoints", f"depth_anything_v2_{args.encoder}.pth")
    if not os.path.isfile(ckpt):
        print(f"[DA2][FATAL] checkpoint not found: {ckpt}")
        sys.exit(1)
    if not os.path.isdir(args.img_dir):
        print(f"[DA2][FATAL] img_dir not found: {args.img_dir}")
        sys.exit(1)

    sys.path.insert(0, args.da2_dir)
    import cv2
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(args.device).eval()

    raw_dir = os.path.join(args.out_dir, "raw_npy_float32")
    os.makedirs(raw_dir, exist_ok=True)

    names = sorted(n for n in os.listdir(args.img_dir)
                   if os.path.splitext(n)[1].lower() in IMG_EXTS)
    if not names:
        print(f"[DA2][FATAL] no images in {args.img_dir}")
        sys.exit(1)

    manifest = {
        "status": "PASS",
        "img_dir": os.path.abspath(args.img_dir).replace("\\", "/"),
        "out_dir": os.path.abspath(args.out_dir).replace("\\", "/"),
        "encoder": args.encoder,
        "checkpoint": os.path.abspath(ckpt).replace("\\", "/"),
        "input_size": int(args.input_size),
        "device": args.device,
        "image_count": len(names),
        "outputs": {"raw_npy_float32": os.path.abspath(raw_dir).replace("\\", "/")},
        "images": [],
    }
    for i, name in enumerate(names):
        path = os.path.join(args.img_dir, name)
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"[DA2][FATAL] failed to read {path}")
            sys.exit(1)
        with torch.no_grad():
            depth = model.infer_image(bgr, args.input_size)        # HxW float32 relative inverse depth
        depth = np.asarray(depth, dtype=np.float32)
        stem = os.path.splitext(name)[0]
        npy_path = os.path.join(raw_dir, stem + ".npy")
        np.save(npy_path, depth)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        manifest["images"].append({
            "image": name,
            "stem": stem,
            "input_sha256": sha,
            "shape_hw": [int(depth.shape[0]), int(depth.shape[1])],
            "raw_npy": os.path.abspath(npy_path).replace("\\", "/"),
            "finite_ratio": float(np.isfinite(depth).mean()),
            "depth_min": float(np.nanmin(depth)),
            "depth_max": float(np.nanmax(depth)),
            "depth_mean": float(np.nanmean(depth)),
        })
        print(f"[DA2] {i + 1}/{len(names)} {name} -> {npy_path}")

    with open(os.path.join(args.out_dir, "depth_anything_v2_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DA2] DONE: {len(names)} images -> {raw_dir}")


if __name__ == "__main__":
    main()
