#!/usr/bin/env bash
# FULL BENCHMARK runner: every scene x every sparse case (12 & 24 train views) in ONE run.
#
# Expected data layout (prebuilt, this runner does NOT build caches):
#   $SCENES_ROOT/<case>/<scene>/<split>/
#     train/{images, sparse/0, pgdr_depth_cache_aligned/{depth_inv_aligned.pt, cfdc_confidence.pt}}
#     test/{images, sparse/0}
#   e.g. _3dgs_splits_lightning_build/mipnerf360_sparse12/kitchen/hold8_train12_sparsegs_triangulate
#        _3dgs_splits_lightning_build/mipnerf360_sparse24/kitchen/hold8_train24_sparsegs_triangulate
# Splits with a missing depth/CFDC cache are SKIPPED with a clear message. CFDC confidence is
# auto-detected by train.py at <depth_cache>/cfdc_confidence.pt (no cache flags needed).
#
# RUN (Lightning, from the repo dir):
#   SCENES_ROOT=/teamspace/studios/this_studio/_3dgs_splits_lightning_build bash tools/run_full_lightning.sh
# Overrides:
#   MODES="cfdc_full_wgsd_b2 hog_hyb"   modes per split (default = baseline + ours, paired in-batch)
#   CASES="mipnerf360_sparse12"         restrict to specific case folders
#   SCENES="kitchen garden"             restrict to specific scenes
#   ITERS=10000  OUTROOT=...  PYTHON=python  EXTRA="--hog_start 100"  DRY_RUN=1 (list only)
set -uo pipefail
if [ -f "./train.py" ]; then CODE="$(pwd)"; elif [ -f "$(dirname "$0")/../train.py" ]; then CODE="$(cd "$(dirname "$0")/.." && pwd)"; else echo "[RUN] run from repo dir"; exit 1; fi
cd "$CODE"
grep -q "cfdc_enable" arguments/__init__.py 2>/dev/null || { echo "[RUN][FATAL] repo missing CFDC/HOG code. Upload the updated repo."; exit 1; }

PYTHON="${PYTHON:-python}"
ITERS="${ITERS:-10000}"
OUTROOT="${OUTROOT:-$CODE/output/full_benchmark}"
MODES="${MODES:-cfdc_full_wgsd_b2 hog_hyb}"
SCENES_ROOT="${SCENES_ROOT:-/teamspace/studios/this_studio/_3dgs_splits_lightning_build}"
CASES="${CASES:-}"
SCENES="${SCENES:-}"
EXTRA="${EXTRA:-}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY="$OUTROOT/FULL_SUMMARY.txt"; mkdir -p "$OUTROOT"
TI="$(seq -s ' ' 1000 1000 "$ITERS")"; [ -n "$TI" ] || TI="$ITERS"

# save PLY + checkpoint at the FINAL iteration only (disk-friendly); summary metrics only (no per-view)
COMMON="--use_existing_split --iterations $ITERS --disable_viewer --quiet \
  --test_iterations $TI --save_iterations $ITERS --checkpoint_iterations $ITERS \
  --metrics_log_interval 1000 --metrics_eval_train_count -1 --metrics_compute_lpips"
CF="--cfdc_enable --cfdc_power 1.0 --cfdc_floor 0.05"
SW="--swd_enable --swd_start 1000 --swd_update_interval 500 --swd_lambda_opacity 0.001"

mode_flags () {
  case "$1" in
    base)      echo "" ;;
    depth)     echo "--use_depth_cache" ;;
    cdw)       echo "--use_depth_cache --cdw_enable --cdw_gamma 2.0" ;;
    cfdc)      echo "--use_depth_cache $CF" ;;
    cfdc_cdw)  echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0" ;;
    cfdc_swd)  echo "--use_depth_cache $CF $SW" ;;
    cfdc_full) echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --dropgaussian_enable --dropgaussian_max_rate 0.1" ;;
    # strongest baseline: static witness-guided dropout
    cfdc_full_wgsd_b2) echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --dropgaussian_enable --dropgaussian_max_rate 0.1 --wgsd_enable --wgsd_beta 2.0" ;;
    # OURS: HOG-GS (held-out depth-harm harvest + static witness hybrid dropout)
    hog_hyb)   echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_signal depth --hog_fold_mode blocked --hog_gamma 1.0 --hog_wgsd_beta 2.0 --hog_meta_interval 50" ;;
    # HOG ablations
    hog_d)     echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_signal depth --hog_fold_mode blocked --hog_gamma 1.0 --hog_meta_interval 50" ;;
    hog_rgb)   echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_signal rgb --hog_fold_mode blocked --hog_gamma 1.0 --hog_meta_interval 50" ;;
    *)         echo "__BAD__" ;;
  esac
}

extract () { "$PYTHON" - "$1" "$ITERS" <<'PY'
import json, os, sys
m, it = sys.argv[1], sys.argv[2]
def f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "NA"
try:
    d = json.load(open(os.path.join(m, "metrics", "run_manifest.json")))
    e = d["metrics"].get(str(it)) or d["metrics"][sorted(d["metrics"], key=int)[-1]]
    print(f"{f(e.get('test_psnr_mean'))} {f(e.get('test_ssim_mean'))} {f(e.get('test_lpips_mean'))} "
          f"{f(e.get('train_psnr_mean'))} {f(e.get('train_ssim_mean'))} {f(e.get('train_lpips_mean'))} "
          f"{e.get('test_view_count')}")
except Exception as ex: print(f"NA NA NA NA NA NA NA ({ex})")
PY
}

echo "=== FULL BENCHMARK === $(date -Iseconds) ITERS=$ITERS ROOT=$SCENES_ROOT MODES=[$MODES] DRY_RUN=$DRY_RUN" | tee -a "$SUMMARY"
FOUND=0
for CASE_DIR in "$SCENES_ROOT"/*/; do
  CASE="$(basename "$CASE_DIR")"
  case "$CASE" in _*) continue ;; esac                       # skip helper/tmp folders
  if [ -n "$CASES" ]; then case " $CASES " in *" $CASE "*) ;; *) continue ;; esac; fi
  for SCENE_DIR in "$CASE_DIR"*/; do
    SCN="$(basename "$SCENE_DIR")"
    if [ -n "$SCENES" ]; then case " $SCENES " in *" $SCN "*) ;; *) continue ;; esac; fi
    for SC in "$SCENE_DIR"*/; do
      [ -d "$SC/train" ] && [ -d "$SC/test" ] || continue
      FOUND=$((FOUND + 1))
      LABEL="$CASE/$SCN"
      CACHE="$SC/train/pgdr_depth_cache_aligned"
      echo "" | tee -a "$SUMMARY"; echo "### $LABEL  ($(basename "$SC"))" | tee -a "$SUMMARY"
      if [ ! -f "$CACHE/depth_inv_aligned.pt" ] || [ ! -f "$CACHE/cfdc_confidence.pt" ]; then
        echo "[SKIP] $LABEL: missing depth/CFDC cache at $CACHE" | tee -a "$SUMMARY"
        continue
      fi
      printf "%-18s %-9s %-9s %-9s %-9s %-9s %-9s %-6s\n" mode tePSNR teSSIM teLPIPS trPSNR trSSIM trLPIPS NViews | tee -a "$SUMMARY"
      for M in $MODES; do
        EX="$(mode_flags "$M")"; [ "$EX" = "__BAD__" ] && { echo "[SKIP] unknown mode $M"; continue; }
        OUT="$OUTROOT/$CASE/$SCN/$M"
        if [ "$DRY_RUN" = "1" ]; then
          printf "%-18s would train -> %s\n" "$M" "$OUT" | tee -a "$SUMMARY"
          continue
        fi
        mkdir -p "$OUT"
        [ -f "$OUT/metrics/run_manifest.json" ] || ( "$PYTHON" train.py -s "$SC" -m "$OUT" $COMMON $EX $EXTRA ) > >(tee "$OUTROOT/$CASE.$SCN.$M.log") 2>&1
        read -r P S L TP TS TL N <<< "$(extract "$OUT")"
        printf "%-18s %-9s %-9s %-9s %-9s %-9s %-9s %-6s\n" "$M" "$P" "$S" "$L" "$TP" "$TS" "$TL" "$N" | tee -a "$SUMMARY"
      done
    done
  done
done
[ "$FOUND" -gt 0 ] || echo "[RUN][FATAL] no splits found under $SCENES_ROOT/<case>/<scene>/<split>" | tee -a "$SUMMARY"
echo "" ; echo "=== DONE === $(date -Iseconds)" | tee -a "$SUMMARY"; cat "$SUMMARY"
