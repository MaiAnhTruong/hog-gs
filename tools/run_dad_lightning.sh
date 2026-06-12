#!/usr/bin/env bash
# DAD (Depth-Anchored Densification) A/B on Lightning. Held-out hold8_train12 sparse split.
#   base  : vanilla 3DGS, NO depth                                   (reference)
#   depth : + aligned depth-reg loss (existing technique)            (the depth baseline to beat)
#   dad   : + depth-reg + Depth-Anchored Densification (OURS)        (split children seeded at depth surface)
#   dad_a06 : dad with a gentler pull (alpha 0.6)                    (ablation)
# Everything else identical. Depth cache auto-loaded from <split>/train/pgdr_depth_cache_aligned.
#
# PRE-REGISTERED bar (vs base AND vs depth, >=2 scenes): test PSNR up, SSIM & LPIPS not worse; ideally PSNR>20.
#
# RUN (from inside the repo):
#   SCENES="/teamspace/studios/this_studio/kitchen/hold8_train12_sparsegs_triangulate" bash tools/run_dad_lightning.sh
#   (multi)  SCENES="<kitchen_split> <garden_split>" ...
set -uo pipefail
if [ -f "./train.py" ]; then CODE="$(pwd)"; elif [ -f "$(dirname "$0")/../train.py" ]; then CODE="$(cd "$(dirname "$0")/.." && pwd)"; else echo "[DAD] run from repo dir"; exit 1; fi
cd "$CODE"
grep -q "dad_enable" arguments/__init__.py 2>/dev/null || { echo "[DAD][FATAL] DAD code missing. Upload updated repo."; exit 1; }

ITERS="${ITERS:-10000}"; OUTROOT="${OUTROOT:-$CODE/output/dad_ab}"; MODES="${MODES:-base depth depth_cfdc depth_swd depth_swd_gate depth_cdw}"
SCENES="${SCENES:-${DATA:-/teamspace/studios/this_studio/kitchen/hold8_train12_sparsegs_triangulate}}"
SUMMARY="$OUTROOT/DAD_SUMMARY.txt"; mkdir -p "$OUTROOT"; TI="$(seq -s ' ' 1000 1000 "$ITERS")"
COMMON="--use_existing_split --iterations $ITERS --disable_viewer --quiet \
  --test_iterations $TI --save_iterations 7000 $ITERS --checkpoint_iterations $ITERS \
  --metrics_log_interval 1000 --metrics_eval_train_count -1 --metrics_eval_per_view --metrics_compute_lpips"

mode_flags () {
  case "$1" in
    base)         echo "" ;;
    depth)        echo "--use_depth_cache" ;;
    depth_cfdc)   echo "--use_depth_cache --cfdc_enable ${CFDC_CACHE:+--cfdc_cache $CFDC_CACHE}" ;;
    depth_cfdc_p2) echo "--use_depth_cache --cfdc_enable --cfdc_power 2.0 ${CFDC_CACHE:+--cfdc_cache $CFDC_CACHE}" ;;
    depth_swd)    echo "--use_depth_cache --cfdc_enable --swd_enable --swd_lambda_opacity 0.001 ${CFDC_CACHE:+--cfdc_cache $CFDC_CACHE}" ;;
    depth_swd_gate) echo "--use_depth_cache --cfdc_enable --swd_enable --swd_birth_gate --swd_lambda_opacity 0.001 ${CFDC_CACHE:+--cfdc_cache $CFDC_CACHE}" ;;
    depth_cdw)    echo "--use_depth_cache --cdw_enable --cdw_gamma 2.0" ;;
    depth_cdw_g3) echo "--use_depth_cache --cdw_enable --cdw_gamma 3.0" ;;
    dad)          echo "--use_depth_cache --dad_enable --dad_alpha 0.8 --dad_agree 0.15" ;;
    *)            echo "__BAD__" ;;
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
echo "=== DAD A/B === $(date -Iseconds) ITERS=$ITERS" | tee -a "$SUMMARY"
for SC in $SCENES; do
  [ -d "$SC/train" ] && [ -d "$SC/test" ] || { echo "[SKIP] $SC has no train/+test/"; continue; }
  SCN="$(basename "$(dirname "$SC")")"
  echo "" | tee -a "$SUMMARY"; echo "### SCENE: $SCN ($SC)" | tee -a "$SUMMARY"
  printf "%-9s %-9s %-9s %-9s %-6s\n" mode PSNR SSIM LPIPS NViews | tee -a "$SUMMARY"
  for M in $MODES; do
    EX="$(mode_flags "$M")"; [ "$EX" = "__BAD__" ] && { echo "[SKIP] $M"; continue; }
    OUT="$OUTROOT/$SCN/$M"; mkdir -p "$OUT"
    [ -f "$OUT/metrics/run_manifest.json" ] || ( python train.py -s "$SC" -m "$OUT" $COMMON $EX ) > >(tee "$OUTROOT/$SCN/$M.log") 2>&1
    read -r P S L N <<< "$(extract "$OUT")"
    printf "%-9s %-9s %-9s %-9s %-6s\n" "$M" "$P" "$S" "$L" "$N" | tee -a "$SUMMARY"
  done
done
echo "" ; echo "=== DONE === $(date -Iseconds)" | tee -a "$SUMMARY"; cat "$SUMMARY"
