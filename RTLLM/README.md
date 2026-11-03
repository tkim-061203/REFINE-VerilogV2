# RTLLM v1.1

The 29 RTLLM v1.1 tasks as used in the paper, one directory per task in `RTLLM/modules/<task>/`:

| File | Content |
|---|---|
| `design_description.txt`, `design_description.RTLLM.txt`, `design_description.xml` | specification in the formats the drivers accept (`--descriptiontype`); `utils/run_heldout.sh` reads `RTLLM.txt` |
| `verified_<task>.v` | reference RTL (hidden from the loop in the held-out protocol) |
| `tb.cpp` | C++/Verilator testbench |
| `category` | task category |

Source benchmark: <https://github.com/hkust-zhiyao/RTLLM> (Lu et al., ASP-DAC 2024).
