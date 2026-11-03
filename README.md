# REFINE-VerilogV2

Code for **REFINE-VerilogV2: Enhancement in Verilog Generation with Small Language Model and EDA Debugging**
(RIVF 2026).

REFINE-VerilogV2 generates Verilog with two LoRA adapters on Qwen2.5-Coder-7B-Instruct (rank r = alpha = 912):
a Generator SLM trained on PyraNet modules that pass a Yosys synthesizability and cell-count filter, and a
Debugger SLM trained on repair pairs. A LangGraph state machine runs Converter (COMBA-PROMPT XML) → Generator →
deterministic Sanitizer → syntax loop (Icarus Verilog / Verilator, TED → Debugger) → testbench loop
(TED → Debugger), with EDTM retry control and coordinated Best-of-N (N = 10).

## Layout

| Path | Content |
|---|---|
| `src/langgraph_core/` | pipeline: graph, prompts, Sanitizer, TED agents, EDTM, Best-of-N |
| `src/benchmark_langgraph.py` | RTLLM driver (trials, jobs) |
| `src/main_langgraph.py` | VerilogEval V2 driver |
| `src/parse_rtllm_trials.py` | syntax / functional pass@k from RTLLM trial reports |
| `src/compute_metrics.py` | ablation table from per-arm results |
| `src/flow_src/` | PyraNet synthesis, cell-count extraction and filtering with Yosys (Table I) |
| `training/` | Generator and Debugger training scripts, logs, adapter configs |
| `utils/run_ablation.sh` | component ablation (Table III) |
| `launch_dual_gpu.sh` | serves the Generator (`:8000`) and the Debugger (`:8001`) with vLLM |
| `RTLLM_v2/` | RTLLM v2.0 tasks as used in the paper |
| `RTLLM/` | RTLLM v1.1 tasks (see `RTLLM/README.md`) |
| `ext/verilog-eval/` | VerilogEval V2 (git submodule) |
| `results/rtllm/` | per-arm pass@5 breakdowns behind Tables II and III |

## Setup

```bash
git clone --recursive https://github.com/<user>/REFINE-VerilogV2.git
cd REFINE-VerilogV2
conda env create -f environment.yaml -n test_VE
conda run -n test_VE pip install -r src/langgraph_core/requirements.txt
```

Tools: Icarus Verilog, Verilator 5.048 (some RTLLM results differ under 5.020), Yosys with the `slang`
frontend (data filter only). Adapters: <https://huggingface.co/nntkim0612>.
Default paths in `launch_dual_gpu.sh`, `Makefile` (`CONDA_ENV`) and `utils/run_ablation.sh` (`PY`) point to
the authors' machine; set them to your own.

## Reproducing the paper

```bash
# serve the two adapters (merged checkpoints)
./launch_dual_gpu.sh --base-model <generator_dir> --merged-model <debugger_dir>

# RTLLM v1.1 and v2.0 (Table II): Best-of-10, 5 trials per task, seed 42
make all_RTLLM COMBA_MAX_SAMPLES=10 COMBA_LLM_SEED=42 SC=1
python src/parse_rtllm_trials.py RTLLM/modules    --out-json rtllm_v1.1_pass5.json
python src/parse_rtllm_trials.py RTLLM_v2/modules --out-json rtllm_v2.0_pass5.json

# VerilogEval V2 (Table II): zero/one-shot x T = 0 / 0.8 (20 samples at T = 0.8)
make VerilogEval VEVAL_CONFIGS='e0_t0 e0_t8 e1_t0 e1_t8' COMBA_MAX_SAMPLES=10 COMBA_LLM_SEED=42 SC=1
grep pass_rate reports/verilogeval/*/summary.txt

# component ablation (Table III): Full, No Debug, No Post, No Both -> reports/abl*/
./utils/run_ablation.sh

# corpus filter (Table I)
make synthesis && make extract && make filter
```

Training is described in `training/README.md`. Runs with the same seed can still differ by 1–2 points
because of vLLM batching.

## Results reported in the paper

| | VerilogEval V2 pass@1 (zero-shot T=0 / T=0.8, one-shot T=0 / T=0.8) | RTLLM v1.1 syntax / func pass@5 | RTLLM v2.0 syntax / func pass@5 |
|---|---|---|---|
| REFINE-VerilogV2 | 77.6 / 78.4 / 76.9 / 78.3 | 100.0 / 96.5 | 100.0 / 84.0 |

| RTLLM | Full | No Debug | No Post | No Both |
|---|---|---|---|---|
| v1.1 func pass@5 | 96.5 | 93.1 | 86.2 | 86.2 |
| v1.1 TB pass rate | 93.8 | 82.1 | 74.5 | 72.4 |
| v2.0 func pass@5 | 84.0 | 84.0 | 76.0 | 76.0 |
| v2.0 TB pass rate | 74.8 | 72.4 | 62.8 | 62.8 |

The RTLLM numbers are `func_pass_at_k` in `results/rtllm/<arm>/*_pass5.json`; the testbench pass rate is the
fraction of passing trials in `func_distribution`.

## Evaluation notes

- The testbench in the repair and selection loop is converted from each benchmark's own reference
  testbench, so these numbers are not directly comparable to single-shot baselines.
- In this protocol the pipeline can read each RTLLM task's reference RTL for the module header and helper
  modules.
- The released Generator was trained with the 156 VerilogEval problems and their reference solutions
  appended to every curriculum stage (`training/scripts/train_gen_new.py`), so its VerilogEval results are
  not held-out.

## Citation

```bibtex
@inproceedings{nguyen2026refineverilogv2,
  author    = {Nguyen, Ngoc-Thien-Kim and Nguyen, Vu-Minh-Thanh and Le, Duc-Hung},
  title     = {{REFINE-VerilogV2}: Enhancement in Verilog Generation with Small Language Model and {EDA} Debugging},
  booktitle = {Proc. RIVF International Conference on Computing and Communication Technologies (RIVF)},
  year      = {2026}
}
```
