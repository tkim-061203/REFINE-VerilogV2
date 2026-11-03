#!/usr/bin/env bash
# run_retrain.sh - Generator retrains of Table IV: clean = filtered corpus,
#                  random = random corpus (training/README.md).
#
#   TRAIN_JSONL=<Pyranet_text_only.jsonl> training/run_retrain.sh clean random   # one after the other
#   SMOKE=1 TRAIN_JSONL=... training/run_retrain.sh clean   # 20 steps per stage, separate _smoke dirs
#
# Per variant: s1 (fresh LoRA on the base) -> s2 -> s3, each stage one epoch,
# loading the previous stage's adapter (training/scripts/train_gen_retrain.py),
# then merge the s3 adapter into a bf16 model under $RETRAIN_DIR/models/.
# Resumable: a stage whose adapter_config.json exists is skipped, a variant
# whose merged model exists is skipped. Needs one free GPU (GPU env below);
# vLLM must not be holding it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TRAIN_PY:-$(command -v python)}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to Pyranet_text_only.jsonl}"; export TRAIN_JSONL
export RETRAIN_DIR="${RETRAIN_DIR:-$HOME/retrain}"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
mkdir -p "$RETRAIN_DIR/logs" "$RETRAIN_DIR/models"
sfx=""; [ "${SMOKE:-0}" = 1 ] && sfx="_smoke"
cd "$RETRAIN_DIR"     # unsloth_compiled_cache lands here, not in the repo

for v in "$@"; do
  merged="$RETRAIN_DIR/models/model_qwen_generator_${v}${sfx}_merged"
  [ -f "$merged/config.json" ] && { echo "skip $v (merged model exists)"; continue; }
  init=""
  for st in s1 s2 s3; do
    ad="$RETRAIN_DIR/adapter_qwen_generator_${v}_${st}${sfx}"
    if [ -f "$ad/adapter_config.json" ]; then echo "skip $v/$st (adapter exists)"; init="$ad"; continue; fi
    echo "=== train $v/$st init=${init:-base} $(date -Iseconds)"
    VARIANT=$v STAGE=$st INIT="$init" "$PY" "$ROOT/training/scripts/train_gen_retrain.py" \
      > "$RETRAIN_DIR/logs/train_${v}_${st}${sfx}.log" 2>&1
    [ -f "$ad/adapter_config.json" ] || { echo "stage $v/$st produced no adapter"; exit 1; }
    init="$ad"
  done
  echo "=== merge $v $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" "$PY" "$ROOT/training/scripts/merge_adapter.py" "$init" "$merged" \
    > "$RETRAIN_DIR/logs/merge_${v}${sfx}.log" 2>&1
  echo "=== done $v -> $merged $(date -Iseconds)"
done
