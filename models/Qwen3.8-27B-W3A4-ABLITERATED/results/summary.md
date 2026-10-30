# Results

All numbers measured on one Radeon AI PRO R9700 (32 GB), ROCm 10, vLLM-based Paiton runtime,
262K context profile, thinking off, 250 W cap, on a KVM guest.

## Quality

MMLU-Pro: 2,800 random items (14 categories × 200, seed 42), 0-shot, temperature 0. Same harness for all three rows.

| Weights | MMLU-Pro |
|---|---:|
| Official W3A4 (upstream, GPTQ) | 57.82% |
| Self-built RTN (abliterated) | 56.68% |
| Self-built GPTQ (abliterated) | **57.86%** |

Other benchmarks (self-built GPTQ):

| Benchmark | Result |
|---|---:|
| GSM8K 5-shot (1,319) | 93.10% |
| HumanEval pass@1 (164) | 90.24% |
| Needle @ 61,440 tokens (160) | 160/160 |

## Speed (BetterBench 0.6.1, 20 passes, single-stream median tok/s)

| Category | Self-built GPTQ | Official 262K |
|---|---:|---:|
| chat | 116.5 | 148.9 |
| code | 163.2 | 202.9 |
| file_edit | 194.0 | 221.3 |
| json | 244.1 | 246.4 |
| math | 197.8 | 211.6 |
| prose | 85.6 | 92.1 |
| reasoning | 109.1 | 131.5 |
| summarization | 153.2 | 142.2 |
| **weighted** | **155.0** | 174.3 |

Concurrency (aggregate tok/s): 130.6 / 236.6 / 332.6 / 436.8 at 1 / 2 / 4 / 8.
