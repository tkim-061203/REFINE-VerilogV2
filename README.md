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
| `src/blind_dataset.py`, `src/heldout_grade.py` | held-out protocol: dataset copy without reference RTL, regrading of final designs (Table IV) |
| `src/langgraph_core/tb_validate.py` | testbench the system writes from the specification, accepted by mutation analysis (F2) |
| `src/heldout_analyze.py` | Table IV and Section IV-D numbers: pass@1/pass@5, task bootstrap, paired contrasts |
| `src/heldout_audit.py`, `src/leakcheck.py` | regrade audit of the original runs, corpus-wide benchmark leak scan |
| `src/flow_src/` | PyraNet synthesis, cell-count extraction and filtering with Yosys (Table I) |
| `training/` | Generator and Debugger training scripts, logs, adapter configs |
| `utils/run_ablation.sh` | component ablation (Table III) |
| `utils/run_heldout.sh` | held-out re-evaluation arms (Table IV) |
| `launch_dual_gpu.sh` | serves the Generator (`:8000`) and the Debugger (`:8001`) with vLLM |
| `RTLLM_v2/` | RTLLM v2.0 tasks as used in the paper |
| `RTLLM/` | RTLLM v1.1 tasks (see `RTLLM/README.md`) |
| `ext/verilog-eval/` | VerilogEval V2 (git submodule) |
| `results/rtllm/` | per-arm pass@5 breakdowns behind Tables II and III |
| `results/heldout/` | expected Table IV / Section IV numbers (`expected.csv`), leak scan (`leaked.json`) |

## Setup

```bash
git clone --recursive https://github.com/tkim-061203/REFINE-VerilogV2.git
cd REFINE-VerilogV2
conda env create -f environment.yaml -n test_VE
conda run -n test_VE pip install -r src/langgraph_core/requirements.txt
```

Tools: Icarus Verilog, Verilator 5.048 (some RTLLM results differ under 5.020), Yosys with the `slang`
frontend (data filter only).

Models (merged bf16, Qwen2.5-Coder-7B-Instruct + LoRA r = alpha = 912):

| Model | Hugging Face |
|---|---|
| Generator (Tables II-IV) | [REFINE-VerilogV2-Generator](https://huggingface.co/nntkim0612/REFINE-VerilogV2-Generator) |
| Debugger (Tables II-IV) | [REFINE-VerilogV2-Debugger](https://huggingface.co/nntkim0612/REFINE-VerilogV2-Debugger) |
| Generator retrained, filtered corpus (Table IV) | [REFINE-VerilogV2-Generator-filtered](https://huggingface.co/nntkim0612/REFINE-VerilogV2-Generator-filtered) |
| Generator retrained, random corpus (Table IV) | [REFINE-VerilogV2-Generator-random](https://huggingface.co/nntkim0612/REFINE-VerilogV2-Generator-random) |
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

### Code revision

Tables I–III were produced with the code of commit `0097ebf` (`git checkout 0097ebf`). Table IV and the
held-out numbers in Sections IV-B, IV-D and IV-E use the current code, which adds the held-out protocol
(`COMBA_BLIND_EVAL`), the generated-testbench path, the regrader and the analysis scripts. With
`COMBA_BLIND_EVAL` unset the current code runs the original protocol, but its prompts and fixes differ from
`0097ebf`.

## Held-out re-evaluation (Table IV)

Every arm hides the reference RTL from the loop (`COMBA_BLIND_EVAL=1` and a dataset copy without
`verified_*.v`), and every final design is regraded against the official testbench in a fresh directory
(RTLLM v1.1 under five fixed stimulus seeds). `utils/run_heldout.sh` starts vLLM with the right checkpoints
for each model, runs the arms and writes `reports/heldout/<model>/<feedback>/<suite>/`.

| Table IV row | command |
|---|---|
| REFINE-VerilogV2, F0 / F2 / F3 | `./utils/run_heldout.sh full F0 F2 F3` |
| F3, base model in both roles | `./utils/run_heldout.sh base F3` |
| F3, single shared adapter | `./utils/run_heldout.sh gen F3` |
| F3, retrained, filtered corpus | `./utils/run_heldout.sh clean F0 F3` |
| F3, retrained, random corpus | `./utils/run_heldout.sh random F3` |

```bash
for m in Generator Debugger Generator-filtered Generator-random; do
  hf download nntkim0612/REFINE-VerilogV2-$m --local-dir $HOME/hf/$m
done
export GEN_CKPT=$HOME/hf/Generator DBG_CKPT=$HOME/hf/Debugger BASE_QWEN=<Qwen2.5-Coder-7B-Instruct dir>
./utils/run_heldout.sh full F0 F2 F3          # suites rl1 rl2 ve by default; about one day per model
./utils/run_heldout.sh base F3
./utils/run_heldout.sh gen F3
./utils/run_heldout.sh full F0s -- rl1 rl2    # no-loop pass@5 (Section IV-B)
./utils/run_heldout.sh full F3nt F3ns         # without TED / without the Sanitizer (Section IV-D)

# retrained Generators, 230,177 rows each (downloaded above, or rebuilt with
# TRAIN_JSONL=<Pyranet_text_only.jsonl> training/run_retrain.sh clean random, see training/README.md)
CLEAN_CKPT=$HOME/hf/Generator-filtered ./utils/run_heldout.sh clean F0 F3
RANDOM_CKPT=$HOME/hf/Generator-random  ./utils/run_heldout.sh random F3

# numbers: arms.csv (p1 with 95% task bootstrap CI, p5), paired.csv (mean per-task difference,
# 95% paired bootstrap CI, 10,000 resamples, seed 0)
python src/heldout_analyze.py reports/heldout --out reports/heldout_results
```

Compare `reports/heldout_results/arms.csv` and `paired.csv` with `results/heldout/expected.csv`, which lists
every Table IV cell and every held-out number quoted in Sections IV-B, IV-D and IV-E with its source column.
RTLLM v1.1 needs the tasks in `RTLLM/modules` (see `RTLLM/README.md`), VerilogEval needs
`ext/verilog-eval`. The corpus-wide leak scan is
`python src/leakcheck.py --out results/heldout/leaked.json --rows-out training/data/leak_rows.json --jobs 32`
(PyraNet-Verilog CSV in the Hugging Face cache).

## Results reported in the paper

| | VerilogEval V2 pass@1 (zero-shot T=0 / T=0.8, one-shot T=0 / T=0.8) | RTLLM v1.1 syntax / func pass@5 | RTLLM v2.0 syntax / func pass@5 |
|---|---|---|---|
| REFINE-VerilogV2 | 77.6 / 78.4 / 76.9 / 78.3 | 100.0 / 96.5 | 100.0 / 84.0 |

Held-out re-evaluation (Table IV, pass@1, %; † Generator trained on the VerilogEval references):

| Configuration | RTLLM v1.1 | RTLLM v2.0 | VerilogEval |
|---|---|---|---|
| REFINE-VerilogV2, F0 (single shot) | 41.4 | 30.0 | 56.4† |
| REFINE-VerilogV2, F2 (oracle-free loop) | 28.3 | 31.6 | 41.0† |
| REFINE-VerilogV2, F3 (official testbench) | 74.5 | 61.2 | 73.7† |
| F3, base model in both roles | 67.6 | 54.0 | 57.1 |
| F3, single shared adapter | 69.7 | 59.2 | 74.4† |
| F3, retrained, filtered corpus | 74.5 | 60.0 | 66.0 |
| F3, retrained, random corpus | 75.2 | 64.4 | 66.7 |

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
  modules. Table IV removes this access and regrades every final design.
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
