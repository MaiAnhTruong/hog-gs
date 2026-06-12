# HOG-GS: Held-Out-Guided Gaussian Regularization for Sparse-View 3D Gaussian Splatting

This repository implements **HOG-GS**, a sparse-view 3DGS method that learns *which Gaussians to
suppress* from views that are **held out of training**: cameras are ring-ordered and partitioned
into rotating blocked folds; the current fold's query views are rendered (never trained on) and the
per-Gaussian opacity gradient of the held-out **depth-prior violation** is harvested as a "harm"
signal. Harm is combined with a static cross-view surface-witness term into per-Gaussian dropout
rates (allocation-only: the mean rate is pinned, so total suppression matches uniform dropout).
The training loss, densification protocol, and evaluation are standard 3DGS; rates apply to
training renders only.

Built on top of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(Inria / MPII) — see [LICENSE.md](LICENSE.md).

## Setup

```bash
pip install plyfile lpips tqdm opencv-python
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
```

Tested with PyTorch >= 2.0 + CUDA. `fused-ssim` is optional (training falls back automatically).

## Data layout

Each scene split is self-contained (no flags needed for the caches; they are auto-detected):

```
<root>/<case>/<scene>/<split>/
  train/
    images/  sparse/0/                          # COLMAP model of the TRAIN views only
    pgdr_depth_cache_aligned/
      depth_inv_aligned.pt                      # [V,1,H,W] COLMAP-scale inverse depth prior
      depth_meta.json                           # per-view affine fits + provenance
      cfdc_confidence.pt                        # cross-view depth-confidence weights
  test/
    images/  sparse/0/                          # held-out evaluation views
```

To build the caches for a new split (Depth-Anything-V2 raw depth -> robust COLMAP-scale alignment
-> cross-view confidence):

```bash
python tools/run_da2_depth.py --img_dir <split>/train/images \
    --out_dir <split>/train/depth_anything_v2_vitl --da2_dir <Depth-Anything-V2 checkout>
python tools/build_depth_cache.py --split_root <split>
python tools/probe_cfdc_depth_consistency.py --split_root <split> \
    --depth_cache <split>/train/pgdr_depth_cache_aligned \
    --out_dir <split>/train/pgdr_depth_cache_aligned \
    --stride 8 --write_confidence --confidence_tau 0.05 --confidence_support 3
```

## Training

Full benchmark (every scene x every sparse case found under the data root):

```bash
SCENES_ROOT=<root> MODES="cfdc_full_wgsd_b2 hog_hyb" bash tools/run_full_lightning.sh
```

Useful overrides: `SCENES="kitchen garden"`, `CASES="mipnerf360_sparse12"`, `ITERS=10000`,
`DRY_RUN=1` (list without training). Results are aggregated in `$OUTROOT/FULL_SUMMARY.txt`.

Single run (HOG-GS on one split):

```bash
python train.py -s <split> -m <out> --use_existing_split --iterations 10000 \
  --use_depth_cache --cfdc_enable --cfdc_power 1.0 --cfdc_floor 0.05 \
  --cdw_enable --cdw_gamma 2.0 \
  --swd_enable --swd_start 1000 --swd_update_interval 500 --swd_lambda_opacity 0.001 \
  --hog_enable --hog_mode grad --hog_signal depth --hog_fold_mode blocked \
  --hog_gamma 1.0 --hog_wgsd_beta 2.0 --hog_meta_interval 50
```

## Modes (ablation ladder of the runner)

| Mode | Meaning |
|---|---|
| `base` | vanilla 3DGS |
| `depth` | + aligned inverse-depth regularization |
| `cdw`, `cfdc`, `cfdc_cdw`, `cfdc_swd` | depth-loss reweighting ablations |
| `cfdc_full` | full evidence stack + uniform DropGaussian |
| `cfdc_full_wgsd_b2` | strongest baseline: static witness-guided dropout |
| `hog_hyb` | **HOG-GS (ours)**: held-out harm + witness hybrid dropout |
| `hog_d`, `hog_rgb` | HOG signal ablations (depth-only / RGB-GT control) |

## Tests

```bash
python tools/hog_v3_unit_test.py        # folds, query weights, harvest sign, probe-not-teacher
python tools/hog_v4_unit_test.py        # depth-signal harm, hybrid exponent, holdout window
python tools/depth_cache_unit_test.py   # robust affine alignment vs synthetic ground truth
```

## License

This project inherits the Gaussian-Splatting License (Inria / MPII, research use) — see
[LICENSE.md](LICENSE.md). `utils/loss_utils.py` contains code from pytorch-ssim (MIT).
