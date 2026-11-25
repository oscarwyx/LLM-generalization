# Qwen3-0.6B Inference Experiments

This repository contains three inference experiments for a team LLM course project. The experiments evaluate KV caching, batch size, and decoding strategy using the published Qwen3-0.6B checkpoint with a LoRA adapter and the 900-question MRQA 2019 SQuAD 1.1 test split from our [team repository](https://github.com/sunweiyi2026-ship-it/LLM-project).

## Notebook

[`inference_experiments.ipynb`](inference_experiments.ipynb) contains the benchmark code and a short analysis.

## KV Cache Results

I ran the same 900 questions three times per setting on an NVIDIA T4 in FP16, using greedy decoding and a batch size of 4.

| Metric | Cache off | Cache on |
| --- | ---: | ---: |
| Throughput (questions/s) | 2.06 | 2.51 |
| Total generation time (s) | 1312.48 | 1074.25 |
| Peak allocated GPU memory (MiB) | 1,349.72 | 1698.24 |
| Peak extra allocated GPU memory (MiB) | 161.54 | 510.05 |
| Exact match (%) | 77.00 | 76.89 |

Caching increased throughput by about **22%** and raised peak extra GPU memory by **348.51 MiB**. The decoded answers matched on **899 of 900 questions**.

![KV cache benchmark results](results-KV%20cache/benchmark.png)

## Batch Size Sweep

I tested batch sizes of 1, 4, 8, and 16 with KV caching enabled and greedy decoding. Each batch size ran three times on the full test split. The figure shows how batch size affects generation throughput and peak allocated GPU memory.

![Generation throughput and peak allocated GPU memory by batch size](results-batch%20size/batch_size_benchmark.png)

## Decoding Strategy

I compared greedy decoding with beam search at widths 2 and 4. Each strategy ran once on the full test split with KV caching enabled and a batch size of 4. The figure compares total generation time and answer quality, measured by exact match (EM) and token-level F1.

![Total generation time, exact match, and F1 by decoding strategy](results-decoding%20strategy/decoding_strategy_benchmark.png)

## Reproduce the Benchmark

1. Open the notebook in Google Colab with a GPU runtime and Python 3.12.
2. Run the notebook cells in order. The benchmark generates the charts and exports the per-batch timings and summary as CSV files.

