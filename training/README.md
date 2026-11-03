# Training

Both adapters use LoRA on all seven attention and feed-forward projections of
`Qwen/Qwen2.5-Coder-7B-Instruct`, r = alpha = 912, dropout 0, bfloat16, 8-bit AdamW, learning rate 2e-5
with a linear schedule, 8192-token context, one epoch.

## Generator (`scripts/train_gen_new.py`)

One RTX 5880 Ada GPU, batch 2, 100 warm-up steps, loss on response tokens only. Training runs as a
curriculum over cell-count stages of the filtered PyraNet corpus (index files from `make filter`); each
stage resumes from the previous checkpoint (`CHECKPOINT` in the script). Logged sizes:

| Stage | Cell range | PyraNet rows | Log |
|---|---|---|---|
| 1 | 6–10 | 85,033 | `logs/train_gen_new.log` |
| 2 | 0–5 | 198,241 | `logs/train_gen_new_0-10.log` |
| 3 | 11–15 | 27,394 | `logs/train_gen_new_0-15.log` |
| 4 | 16+ | 5,225 | `logs/train_gen_new_0-35.log` |

Total: 315,893 PyraNet rows. Every stage also concatenates `VE_text_156.jsonl`, the 156 VerilogEval problems
with their reference solutions; drop it for a model that is held-out on VerilogEval.

## Debugger (`scripts/train_debugger_2gpu.py`)

Model split over two GPUs, micro-batch 1, gradient accumulation 4, 3% warm-up. Data: 107,729 repair pairs
(39,203 from Generator failures on the training corpus, 68,526 from synthetic error injection), no
VerilogEval files.

## Merging

`scripts/merge_lora.py` merges an adapter into the base model for vLLM serving. Adapter configs and trainer
states of the released checkpoints are in `metadata/`.
