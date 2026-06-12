#!/usr/bin/env bash
# CFDC-GS A/B on Lightning. Held-out hold8_train12 sparse split. Soft, confidence-aware depth.
#   base       : vanilla 3DGS, no depth                                       (reference)
#   depth      : + aligned soft depth-reg loss (existing technique)           (the baseline to beat, ~19.13)
#   cdw        : depth + texture-complementary reweighting (CDW)
#   cfdc       : depth + CROSS-VIEW depth-confidence reweighting (CFDC, OURS)  (downweights cross-view-
#                inconsistent depth pixels - 25% on kitchen - instead of trusting all depth uniformly)
#   cfdc_cdw   : depth + CFDC + CDW (geometry confidence x texture)
#   cfdc_swd   : depth + CFDC + SWD soft opacity penalty on cross-view-confirmed FRONT floaters
#   cfdc_full  : depth + CFDC + CDW + SWD + DropGaussian
# All SOFT (no hard placement, no prune, no test view). CFDC cache (cfdc_confidence.pt) via $CFDC_CACHE.
#
# PRE-REGISTERED bar (vs base AND vs depth): test PSNR up, SSIM & LPIPS not worse; goal PSNR>20.
#
# RUN (from repo dir), single scene with its CFDC cache:
#   CFDC_CACHE=/teamspace/studios/this_studio/cfdc_kitchen_cache_s8 \
#   SCENES="/teamspace/studios/this_studio/kitchen/hold8_train12_sparsegs_triangulate" bash tools/run_cfdc_lightning.sh
set -uo pipefail
if [ -f "./train.py" ]; then CODE="$(pwd)"; elif [ -f "$(dirname "$0")/../train.py" ]; then CODE="$(cd "$(dirname "$0")/.." && pwd)"; else echo "[CFDC] run from repo dir"; exit 1; fi
cd "$CODE"
grep -q "cfdc_enable" arguments/__init__.py 2>/dev/null || { echo "[CFDC][FATAL] CFDC code missing. Upload updated repo."; exit 1; }

ITERS="${ITERS:-10000}"; OUTROOT="${OUTROOT:-$CODE/output/cfdc_ab}"
MODES="${MODES:-base depth cdw cfdc cfdc_cdw cfdc_swd cfdc_full}"
SCENES="${SCENES:-${DATA:-/teamspace/studios/this_studio/kitchen/hold8_train12_sparsegs_triangulate}}"
CFDC_CACHE="${CFDC_CACHE:-$CODE/experiments/cfdc_kitchen_cache_s8}"
SUMMARY="$OUTROOT/CFDC_SUMMARY.txt"; mkdir -p "$OUTROOT"; TI="$(seq -s ' ' 1000 1000 "$ITERS")"
[ -f "$CFDC_CACHE/cfdc_confidence.pt" ] || echo "[CFDC][WARN] no cfdc_confidence.pt at $CFDC_CACHE -> cfdc modes inactive"

COMMON="--use_existing_split --iterations $ITERS --disable_viewer --quiet \
  --test_iterations $TI --save_iterations 7000 $ITERS --checkpoint_iterations $ITERS \
  --metrics_log_interval 1000 --metrics_eval_train_count -1 --metrics_eval_per_view --metrics_compute_lpips"
CF="--cfdc_enable --cfdc_cache $CFDC_CACHE --cfdc_power 1.0 --cfdc_floor 0.05"
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
    # --- attribution ablations (isolate what drives the cfdc_full gain) ---
    depth_drop)     echo "--use_depth_cache --dropgaussian_enable --dropgaussian_max_rate 0.1" ;;
    cfdc_nodrop)    echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW" ;;
    depth_swd_bg)   echo "--use_depth_cache $CF $SW --swd_birth_gate" ;;
    # --- push higher + strengthen the NOVEL cross-view surface-witness (birth gate) + tuning ---
    cfdc_full_bg)   echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --swd_birth_gate --dropgaussian_enable --dropgaussian_max_rate 0.1" ;;
    cfdc_full_d2)   echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --dropgaussian_enable --dropgaussian_max_rate 0.2" ;;
    cfdc_full_bg_d2) echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --swd_birth_gate --dropgaussian_enable --dropgaussian_max_rate 0.2" ;;
    # --- WGSD: witness-guided dropout (improves DropGaussian's stated limitations; OUR mechanism) ---
    cfdc_full_wgsd)    echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --dropgaussian_enable --dropgaussian_max_rate 0.1 --wgsd_enable --wgsd_beta 1.0" ;;
    cfdc_full_wgsd_b2) echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --dropgaussian_enable --dropgaussian_max_rate 0.1 --wgsd_enable --wgsd_beta 2.0" ;;
    depth_drop_wgsd)   echo "--use_depth_cache $CF $SW --dropgaussian_enable --dropgaussian_max_rate 0.1 --wgsd_enable --wgsd_beta 1.0" ;;
    # --- ECU: LEARNED evidence-conditioned dropout head (vs ugod_style = same net, shape-only inputs) ---
    ecu)         echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --ecu_enable --ecu_inputs evidence" ;;
    ugod_style)  echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --ecu_enable --ecu_inputs shape" ;;
    ecu_both)    echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --ecu_enable --ecu_inputs both" ;;
    # --- HOG-GS: held-out-guided policy (OURS, principle-level). Same stack; dropout head learned
    #     by rotating support/query outer ES instead of fixed mapping / same-view training. ---
    # v3 grad-mode (first-order heldout harm + blocked folds); _g2 = sharper mapping; _il = interleaved-fold ablation
    hog)         echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_fold_mode blocked --hog_gamma 1.0 --hog_meta_interval 50" ;;
    hog_g2)      echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_fold_mode blocked --hog_gamma 2.0 --hog_meta_interval 50" ;;
    hog_il)      echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_fold_mode interleaved --hog_gamma 1.0 --hog_meta_interval 50" ;;
    hog_es)      echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode es --hog_fold_mode blocked" ;;
    # v4: held-out DEPTH-PRIOR harm (external-anchored signal; corr_front expected POSITIVE)
    hog_d)       echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_signal depth --hog_fold_mode blocked --hog_gamma 1.0 --hog_meta_interval 50" ;;
    hog_d_g2)    echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_signal depth --hog_fold_mode blocked --hog_gamma 2.0 --hog_meta_interval 50" ;;
    hog_hyb)     echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_signal depth --hog_fold_mode blocked --hog_gamma 1.0 --hog_wgsd_beta 2.0 --hog_meta_interval 50" ;;
    # v5 HOG-P: hog_hyb + persistent robust harm field. persist: EMA harm inherited through
    # densification, rates RE-DERIVED when N changes (legacy ran UNIFORM ~49% of iters: end-of-iter
    # densify at t=0 mod 100 invalidated the same-iteration harvest). median: per-view standardize +
    # unweighted median across query views (one exploding L_q view cannot flip the allocation).
    hog_p)       echo "--use_depth_cache $CF --cdw_enable --cdw_gamma 2.0 $SW --hog_enable --hog_mode grad --hog_signal depth --hog_fold_mode blocked --hog_gamma 1.0 --hog_wgsd_beta 2.0 --hog_meta_interval 50 --hog_persist --hog_median --hog_eta 0.3" ;;
    *)         echo "__BAD__" ;;
  esac
}
extract () { python - "$1" "$ITERS" <<'PY'
import json, os, sys
m, it = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(os.path.join(m, "metrics", "run_manifest.json")))
    e = d["metrics"].get(str(it)) or d["metrics"][sorted(d["metrics"], key=int)[-1]]
    print(f"{e.get('test_psnr_mean')} {e.get('test_ssim_mean')} {e.get('test_lpips_mean')} {e.get('test_view_count')}")
except Exception as ex: print(f"NA NA NA NA ({ex})")
PY
}
echo "=== CFDC-GS A/B === $(date -Iseconds) ITERS=$ITERS CFDC_CACHE=$CFDC_CACHE" | tee -a "$SUMMARY"
for SC in $SCENES; do
  [ -d "$SC/train" ] && [ -d "$SC/test" ] || { echo "[SKIP] $SC no train/+test/"; continue; }
  SCN="$(basename "$(dirname "$SC")")"
  echo "" | tee -a "$SUMMARY"; echo "### SCENE: $SCN" | tee -a "$SUMMARY"
  printf "%-10s %-9s %-9s %-9s %-6s\n" mode PSNR SSIM LPIPS NViews | tee -a "$SUMMARY"
  for M in $MODES; do
    EX="$(mode_flags "$M")"; [ "$EX" = "__BAD__" ] && { echo "[SKIP] $M"; continue; }
    OUT="$OUTROOT/$SCN/$M"; mkdir -p "$OUT"
    [ -f "$OUT/metrics/run_manifest.json" ] || ( python train.py -s "$SC" -m "$OUT" $COMMON $EX ) > >(tee "$OUTROOT/$SCN/$M.log") 2>&1
    read -r P S L N <<< "$(extract "$OUT")"
    printf "%-10s %-9s %-9s %-9s %-6s\n" "$M" "$P" "$S" "$L" "$N" | tee -a "$SUMMARY"
  done
done
echo "" ; echo "=== DONE === $(date -Iseconds)" | tee -a "$SUMMARY"; cat "$SUMMARY"
