#!/usr/bin/env bash
# ============================================================================
# run_heldout.sh - held-out re-evaluation (Table IV, Sections IV-B and IV-D).
#
#   ./utils/run_heldout.sh <model> [feedback ...] [-- suites]
#
#   model    : which merged checkpoints vLLM serves (local directories)
#     full    released Generator (GEN_CKPT) + released Debugger (DBG_CKPT)
#     gen     released Generator in both roles (single shared adapter)
#     base    Qwen2.5-Coder-7B-Instruct in both roles (BASE_QWEN)
#     clean   Generator retrained on the filtered corpus (CLEAN_CKPT) + DBG_CKPT
#     random  Generator retrained on the random corpus (RANDOM_CKPT) + DBG_CKPT
#   feedback : default F0 F2 F3
#     F0    single shot: 1 sample, no TED / Debugger, no testbench in the loop
#     F2    Best-of-10 + repair against a testbench the system writes from the
#           specification, accepted by mutation analysis (oracle-free); one
#           testbench per problem and trial, shared by all samples
#     F3    official testbench in the loop, reference RTL hidden
#     F0s   F0 at temperature 0.8, 10 independent trials (RTLLM) or 20 samples
#           (VerilogEval): the no-loop pass@5 of Section IV-B
#     F3nt  F3 without TED, F3ns F3 without the Sanitizer (Section IV-D)
#   suites   : rl1 (RTLLM v1.1) rl2 (RTLLM v2.0) ve (VerilogEval V2), default all
#
# Every arm hides the reference RTL (verified_*.v) from the loop
# (COMBA_BLIND_EVAL=1 + a blind copy of the dataset), and every final design is
# regraded against the official testbench in a fresh directory by
# src/heldout_grade.py (RTLLM; v1.1 under five fixed seeds) or by the
# VerilogEval final check. Arms are resumable: a finished arm leaves DONE and
# is skipped on rerun. Output: $OUT/<model>/<feedback>/<suite>/, then
#   python src/heldout_analyze.py reports/heldout --out reports/heldout_results
#
# Example (2x 48 GB GPUs, about one day per model):
#   GEN_CKPT=... DBG_CKPT=... ./utils/run_heldout.sh full
#   GEN_CKPT=... DBG_CKPT=... ./utils/run_heldout.sh full F3nt F3ns -- rl1 rl2
# ============================================================================
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$(command -v python)}"
OUT="${OUT:-$ROOT/reports/heldout}"
JOBS="${JOBS:-10}"
TRIALS="${TRIALS:-5}"
TRIALS_F0S="${TRIALS_F0S:-10}"   # independent samples per RTLLM task for F0s
BASE_QWEN="${BASE_QWEN:-}"       # local dir of Qwen2.5-Coder-7B-Instruct (launch_dual_gpu.sh checks -d)
GEN_CKPT="${GEN_CKPT:-}"         # local dir of the merged released Generator
DBG_CKPT="${DBG_CKPT:-}"         # local dir of the merged released Debugger
CLEAN_CKPT="${CLEAN_CKPT:-}"     # merged Generator from training/run_retrain.sh clean
RANDOM_CKPT="${RANDOM_CKPT:-}"   # merged Generator from training/run_retrain.sh random
cd "$ROOT"
export PATH="$(dirname "$PY"):$PATH"   # launch_dual_gpu.sh runs bare `python -m vllm...`

MODEL="${1:?model: full|gen|base|clean|random}"; shift
FEEDBACK=(); SUITES=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; SUITES=("$@"); break; fi
  FEEDBACK+=("$1"); shift
done
[ ${#FEEDBACK[@]} -eq 0 ] && FEEDBACK=(F0 F2 F3)
[ ${#SUITES[@]} -eq 0 ] && SUITES=(rl1 rl2 ve)

need_dir () { [ -d "$2" ] || { echo "set $1 to a local model directory"; exit 1; }; }
case "$MODEL" in
  full)   need_dir GEN_CKPT "$GEN_CKPT"; need_dir DBG_CKPT "$DBG_CKPT"
          GEN="$GEN_CKPT"; DBG="$DBG_CKPT" ;;
  gen)    need_dir GEN_CKPT "$GEN_CKPT"
          GEN="$GEN_CKPT"; DBG="$GEN_CKPT" ;;     # Generator adapter also plays Debugger
  base)   need_dir BASE_QWEN "$BASE_QWEN"
          GEN="$BASE_QWEN"; DBG="$BASE_QWEN" ;;
  clean)  need_dir CLEAN_CKPT "$CLEAN_CKPT"; need_dir DBG_CKPT "$DBG_CKPT"
          GEN="$CLEAN_CKPT"; DBG="$DBG_CKPT" ;;
  random) need_dir RANDOM_CKPT "$RANDOM_CKPT"; need_dir DBG_CKPT "$DBG_CKPT"
          GEN="$RANDOM_CKPT"; DBG="$DBG_CKPT" ;;
  *) echo "unknown model $MODEL (full|gen|base|clean|random)"; exit 1 ;;
esac
for fb in "${FEEDBACK[@]}"; do
  case "$fb" in F0|F0s|F2|F3|F3nt|F3ns) ;; *) echo "unknown feedback $fb"; exit 1 ;; esac
done
for suite in "${SUITES[@]}"; do
  case "$suite" in rl1|rl2|ve) ;; *) echo "unknown suite $suite"; exit 1 ;; esac
done

# ---- serve the right checkpoints, then record what is actually served ------
serve () {
  bash ./launch_dual_gpu.sh --restart --base-model "$GEN" --merged-model "$DBG"   # works without the +x bit
  for i in $(seq 1 120); do
    if curl -sf -m3 localhost:8000/health >/dev/null && curl -sf -m3 localhost:8001/health >/dev/null; then
      return 0; fi
    sleep 10
  done
  echo "vLLM did not come up"; exit 1
}
provenance () {   # $1 = arm dir
  { echo "model_tag: $MODEL"; echo "expected_gen: $GEN"; echo "expected_dbg: $DBG"
    echo "served_gen: $(curl -s localhost:8000/v1/models)"; echo "served_dbg: $(curl -s localhost:8001/v1/models)"
    echo "git: $(git rev-parse HEAD 2>/dev/null) dirty=$(git status --porcelain -uno 2>/dev/null | wc -l)"
    echo "date: $(date -Iseconds)"
    echo "arm_env: $COMMON $(feedback_env "$fb")"
    { env | grep -E '^COMBA_' || true; } | sort; } > "$1/provenance.txt"   # grep finds nothing -> exit 1 under pipefail
}
vllm_tokens () {   # $1 port -> "prompt completion" token totals since the server started
  local m; m=$(curl -sf -m5 "localhost:$1/metrics" 2>/dev/null) || m=""
  [ -n "$m" ] || { echo "-1 -1"; return 0; }
  printf '%s\n' "$m" | awk '/^vllm:prompt_tokens_total/{p+=$NF} /^vllm:generation_tokens_total/{g+=$NF} END{printf "%d %d\n", p, g}'
}
write_tokens () {   # $1 arm, $2 $3 = gen before/after, $4 $5 = dbg before/after ("prompt completion")
  "$PY" - "$1/tokens.json" "$2" "$3" "$4" "$5" <<'EOF'
import json, sys
out, g0, g1, d0, d1 = sys.argv[1:6]
def delta(a, b):
    a, b = [int(x) for x in a.split()], [int(x) for x in b.split()]
    return None if min(a + b) < 0 else {"prompt": b[0] - a[0], "completion": b[1] - a[1]}
json.dump({"generator": delta(g0, g1), "debugger": delta(d0, d1),
           "source": "vLLM /metrics counters, delta over the arm"}, open(out, "w"), indent=2)
EOF
}
export VLLM_LOG_STATS=1   # launch_dual_gpu.sh keeps vLLM stats on, so /metrics counts tokens
if pgrep -af 'benchmark_langgraph\.py' >/dev/null; then echo "another benchmark is running"; exit 1; fi
serve

COMMON="COMBA_BLIND_EVAL=1 COMBA_LLM_SEED=42 COMBA_XML_ESCAPE=1 COMBA_PIPELINE_TIMEOUT=600 \
COMBA_WALL_BUDGET=1200 COMBA_SC_MEMORY=0 COMBA_SC_VOTE=0 COMBA_QUIET=1"

feedback_env () {   # $1 = feedback ; echoes env assignments
  case "$1" in
    F0)   echo "COMBA_SELF_CONSISTENCY=0 COMBA_MAX_SAMPLES=1 COMBA_USE_TED=0 COMBA_USE_DEBUGGER_SLM=0 COMBA_SKIP_TB_IF_NO_GOLDEN=1" ;;
    F0s)  echo "COMBA_SELF_CONSISTENCY=0 COMBA_MAX_SAMPLES=1 COMBA_USE_TED=0 COMBA_USE_DEBUGGER_SLM=0 COMBA_SKIP_TB_IF_NO_GOLDEN=1 COMBA_TEMPERATURE=0.8" ;;
    F2)   echo "COMBA_SELF_CONSISTENCY=1 COMBA_MAX_SAMPLES=10 COMBA_SKIP_TB_IF_NO_GOLDEN=0 COMBA_TB_VALIDATE=1 COMBA_TS_SIMULATOR=iverilog" ;;
    F3)   echo "COMBA_SELF_CONSISTENCY=1 COMBA_MAX_SAMPLES=10 COMBA_SKIP_TB_IF_NO_GOLDEN=0 COMBA_TS_SIMULATOR=verilator" ;;
    F3nt) echo "COMBA_SELF_CONSISTENCY=1 COMBA_MAX_SAMPLES=10 COMBA_SKIP_TB_IF_NO_GOLDEN=0 COMBA_TS_SIMULATOR=verilator COMBA_USE_TED=0" ;;
    F3ns) echo "COMBA_SELF_CONSISTENCY=1 COMBA_MAX_SAMPLES=10 COMBA_SKIP_TB_IF_NO_GOLDEN=0 COMBA_TS_SIMULATOR=verilator COMBA_USE_SANITIZER=0" ;;
  esac
}
blind_mode () { case "$1" in F3*) echo tb ;; *) echo spec ;; esac; }   # F3*: official TB kept, else spec only

run_rtllm () {   # $1 suite rl1|rl2, $2 feedback, $3 arm dir
  local suite="$1" fb="$2" arm="$3" src desc extra trials
  if [ "$suite" = rl1 ]; then src=RTLLM/modules; desc=RTLLM.txt; extra="";
  else src=RTLLM_v2/modules; desc=txt; extra="COMBA_FORCE_XML=1"; fi
  trials=$TRIALS
  case "$fb" in F0) trials=1 ;; F0s) trials=$TRIALS_F0S ;; esac   # F0 is low-temperature seeded, one draw suffices
  local blind="$arm/RTLLM_${suite}_$(blind_mode "$fb")/modules"
  "$PY" src/blind_dataset.py "$src" "$blind" --mode "$(blind_mode "$fb")" | tee "$arm/blind.log"
  env $COMMON $(feedback_env "$fb") $extra COMBA_GENTB_CACHE_DIR="$arm/gentb_cache" \
    "$PY" src/benchmark_langgraph.py --modules-dir "$blind" --descriptiontype "$desc" \
      --trials "$trials" --jobs "$JOBS" --output-dir "$arm/fixrate"
  "$PY" src/parse_rtllm_trials.py "$blind" --out-json "$arm/inloop_breakdown.json" || true
  env COMBA_TS_SIMULATOR=verilator COMBA_BLIND_EVAL=1 COMBA_QUIET=1 \
    "$PY" src/heldout_grade.py --runs "$blind" --official "$src" --out-json "$arm/heldout.json" --jobs 8
}

run_ve () {   # $1 feedback, $2 arm dir
  local fb="$1" arm="$2" sc=1 n=10 cfg=e0_t0
  case "$fb" in F0) sc=0; n=1 ;; F0s) sc=0; n=1; cfg=e0_t8 ;; esac   # e0_t8 = 20 samples at T=0.8
  local inloop=""
  case "$fb" in F3*) ;; *) mkdir -p "$arm/ve_blind"; inloop="COMBA_INLOOP_DATASET_DIR=$arm/ve_blind" ;; esac
  env $COMMON $(feedback_env "$fb") $inloop COMBA_GENTB_CACHE_DIR="$arm/gentb_cache" \
    make VerilogEval VEVAL_CONFIGS=$cfg SC=$sc COMBA_MAX_SAMPLES=$n LANGGRAPH_JOBS="$JOBS" \
      VEVAL_BUILD_ROOT="$arm/ve_build" VEVAL_SWEEP_REPORTS="$arm/ve_reports"
}

for fb in "${FEEDBACK[@]}"; do
  for suite in "${SUITES[@]}"; do
    arm="$OUT/$MODEL/$fb/$suite"
    if [ -f "$arm/DONE" ]; then echo "skip $arm"; continue; fi
    rm -rf "$arm"; mkdir -p "$arm"
    echo "=========== $MODEL / $fb / $suite  ($(date)) ==========="
    provenance "$arm"
    t0=$(date +%s); tg0=$(vllm_tokens 8000); td0=$(vllm_tokens 8001)
    case "$suite" in
      ve)      run_ve "$fb" "$arm" ;;
      rl1|rl2) run_rtllm "$suite" "$fb" "$arm" ;;
    esac
    echo "elapsed_s: $(( $(date +%s) - t0 ))" >> "$arm/provenance.txt"
    write_tokens "$arm" "$tg0" "$(vllm_tokens 8000)" "$td0" "$(vllm_tokens 8001)" || true
    touch "$arm/DONE"
  done
done
echo "done: $OUT/$MODEL"
