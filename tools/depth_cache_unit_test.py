"""Unit test for tools/build_depth_cache.py against a SYNTHETIC ground truth.

Builds a fake split (train/images + train/sparse/0 COLMAP model) where the raw depth maps are
generated from a KNOWN affine relation  raw = (inv_colmap - b) / a  plus outliers on a subset of
observations, AND the raw maps live at HALF the COLMAP camera resolution (the measured real-data
trap: DA2 raws generated from 1600-wide images while COLMAP ran at native resolution - sampling
without coordinate rescale collapses the fit). The builder must (1) recover (a, b) per view
despite the outliers and the resolution mismatch, (2) write a [V,1,H,W] tensor equal to
a*resized_raw+b, (3) produce a depth_meta.json that utils.depthguide.load_aligned_depth_cache
accepts and matches to cameras by stem.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.read_write_model import Camera, Image, Point3D, write_model, qvec2rotmat

rng = np.random.default_rng(0)
ROOT = tempfile.mkdtemp(prefix="depth_cache_test_")
TRAIN = os.path.join(ROOT, "train")
os.makedirs(os.path.join(TRAIN, "images"))
os.makedirs(os.path.join(TRAIN, "sparse", "0"))

W_IMG, H_IMG = 320, 200          # COLMAP camera / image resolution (below 1600 -> no train resize)
W_RAW, H_RAW = 160, 100          # raw depth at HALF resolution (coordinate-rescale path)
SX, SY = W_RAW / W_IMG, H_RAW / H_IMG
V = 3
TRUE = [(0.004, 0.12), (0.006, 0.20), (0.005, 0.15)]      # per-view (a, b)

# cameras on a small arc looking at the point cloud around z ~ [4, 8]
pts = rng.uniform(-1.5, 1.5, size=(300, 3)) + np.array([0.0, 0.0, 6.0])
cams, imgs, p3d = {}, {}, {}
fx = 300.0
for pid, xyz in enumerate(pts, start=1):
    p3d[pid] = Point3D(id=pid, xyz=xyz, rgb=np.array([128, 128, 128]),
                       error=np.array(0.5), image_ids=np.array([1]), point2D_idxs=np.array([0]))

from PIL import Image as PILImage
for v in range(V):
    cams[v + 1] = Camera(id=v + 1, model="PINHOLE", width=W_IMG, height=H_IMG,
                         params=np.array([fx, fx, W_IMG / 2.0, H_IMG / 2.0]))
    ang = (v - 1) * 0.1
    qvec = np.array([math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0])    # small yaw
    tvec = np.array([0.1 * v, 0.0, 0.0])
    R = qvec2rotmat(qvec)
    xys, pids = [], []
    a, b = TRUE[v]
    raw = rng.uniform(5.0, 40.0, size=(H_RAW, W_RAW)).astype(np.float32)  # background raw values
    used_cells = set()
    for pid, X in p3d.items():
        pc = R @ X.xyz + tvec
        z = pc[2]
        if z <= 0.1:
            continue
        u = fx * pc[0] / z + W_IMG / 2.0                 # COLMAP (camera-resolution) coords
        w_ = fx * pc[1] / z + H_IMG / 2.0
        ur, wr = u * SX, w_ * SY                          # raw-resolution coords
        if not (1 <= ur < W_RAW - 2 and 1 <= wr < H_RAW - 2):
            continue
        # one observation per 4x4 raw cell so the written 2x2 patches never overlap/contaminate
        cell = (int(wr) // 4, int(ur) // 4)
        if cell in used_cells:
            continue
        used_cells.add(cell)
        inv = 1.0 / z
        raw_val = (inv - b) / a                       # ground-truth affine relation
        if len(xys) % 5 == 4:                         # 20% gross outliers
            raw_val += rng.uniform(50, 100)
        # write a constant 2x2 patch at RAW resolution so bilinear sampling returns raw_val exactly
        r0, c0 = int(wr), int(ur)
        raw[r0:r0 + 2, c0:c0 + 2] = raw_val
        xys.append([u, w_])                           # stored in COLMAP camera coords
        pids.append(pid)
    assert len(xys) > 40, f"view {v}: too few synthetic observations ({len(xys)})"
    name = f"view{v:02d}.png"
    PILImage.new("RGB", (W_IMG, H_IMG)).save(os.path.join(TRAIN, "images", name))
    np.save(os.path.join(TRAIN, "raws_" + str(v) + ".npy"), raw)          # staged below
    imgs[v + 1] = Image(id=v + 1, qvec=qvec, tvec=tvec, camera_id=v + 1, name=name,
                        xys=np.array(xys), point3D_ids=np.array(pids))

write_model(cams, imgs, p3d, os.path.join(TRAIN, "sparse", "0"), ext=".bin")
raw_dir = os.path.join(TRAIN, "depth_anything_v2_vitl", "raw_npy_float32")
os.makedirs(raw_dir)
for v in range(V):
    shutil.move(os.path.join(TRAIN, f"raws_{v}.npy"), os.path.join(raw_dir, f"view{v:02d}.npy"))

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = subprocess.run([sys.executable, os.path.join(repo, "tools", "build_depth_cache.py"),
                    "--split_root", ROOT], capture_output=True, text=True)
print(r.stdout.strip())
assert r.returncode == 0, f"builder failed:\n{r.stderr}"

import cv2
cache = os.path.join(TRAIN, "pgdr_depth_cache_aligned")
meta = json.load(open(os.path.join(cache, "depth_meta.json")))
assert meta["schema"] == "PGDR_DEPTH_CACHE_V2_ALIGNED" and meta["status"] == "PASS"
tensor = torch.load(os.path.join(cache, "depth_inv_aligned.pt"))
assert list(tensor.shape) == [V, 1, H_IMG, W_IMG], f"bad tensor shape {list(tensor.shape)}"

for v, fr in enumerate(meta["frames"]):
    a_t, b_t = TRUE[v]
    da = abs(fr["fit_a"] - a_t) / a_t
    db = abs(fr["fit_b"] - b_t) / b_t
    assert fr["fit_scope"].startswith("per_view"), f"view {v} fell back: {fr['fit_scope']}"
    assert da < 0.02 and db < 0.02, f"view {v}: fit (a={fr['fit_a']:.6f}, b={fr['fit_b']:.4f}) " \
                                    f"vs true ({a_t}, {b_t}) rel err a={da:.3f} b={db:.3f}"
    raw = np.load(os.path.join(raw_dir, f"view{v:02d}.npy"))
    resized = cv2.resize(raw, (W_IMG, H_IMG), interpolation=cv2.INTER_LINEAR)
    expect = np.clip(fr["fit_a"] * resized.astype(np.float64) + fr["fit_b"], 0.0, None)
    got = tensor[v, 0].numpy()
    assert np.allclose(got, expect, atol=1e-5), f"view {v}: aligned map mismatch"
print(f"[ok] robust fit recovers (a,b) within 2% despite 20% gross outliers AND half-resolution "
      f"raws (xys rescale); tensor == a*resize(raw)+b exactly")

# loader contract: load_aligned_depth_cache matches by stem and resizes to camera resolution
from utils.depthguide import load_aligned_depth_cache

class FakeCam:
    def __init__(self, name, h, w):
        self.image_name = name
        self.image_height = h
        self.image_width = w

if torch.cuda.is_available():
    fake = [FakeCam(f"view{v:02d}.png", H_IMG // 2, W_IMG // 2) for v in range(V)]   # force resize
    n = load_aligned_depth_cache(fake, cache)
    assert n == V and all(c.depth_reliable for c in fake)
    assert tuple(fake[0].invdepthmap.shape) == (H_IMG // 2, W_IMG // 2)
    print("[ok] depthguide loader consumes the cache (stem match + resize to camera resolution)")
else:
    print("[skip] loader test (no CUDA)")

shutil.rmtree(ROOT, ignore_errors=True)
print("\nALL DEPTH-CACHE UNIT TESTS PASSED")
