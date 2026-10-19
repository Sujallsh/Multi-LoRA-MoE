# Task Two: Multi-LoRA for Multi-Task Learning

This project implements a CPU-friendly Multi-LoRA Mixture-of-Experts classifier for three open-source NLP tasks. It uses a frozen tiny BERT encoder, one low-rank feature adapter per task, a sparse learned router, and task-specific classification heads.

## Architecture

```text
Text -> frozen BERT -> pooled feature -> LoRA experts -> MoE router -> task head
```

## Setup on Windows

```powershell
cd C:\NPU\NPU\NPU-2026-sem7\homework_task2
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick smoke test

Run this first. It uses small samples and verifies the complete pipeline:

```powershell
.\.venv\Scripts\python.exe train.py --train-samples 200 --eval-samples 100 --batch-size 16 --expert-epochs 1 --router-epochs 1 --joint-epochs 0 --output-dir outputs/smoke
```

## Main CPU experiment

```powershell
.\.venv\Scripts\python.exe train.py --train-samples 3000 --eval-samples 500 --batch-size 16 --expert-epochs 1 --router-epochs 2 --joint-epochs 1 --output-dir outputs/cpu_demo
```

The first run downloads `google/bert_uncased_L-2_H-128_A-2`, AG News, Emotion, and DBpedia 14 from Hugging Face. Later runs use the local cache.

## Prediction

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoint outputs/cpu_demo/model.pt --task emotion --text "I finally passed the exam and I am thrilled"
```

The output contains the numeric label, confidence, and weight assigned to each expert.

## Results

Training creates:

- `model.pt`: configuration and trained weights.
- `training_history.json`: losses and validation metrics.
- `comparison.json`: uniform, learned MoE, and task-expert results.
- `tokenizer/`: tokenizer configuration.

Use `comparison.json` to complete the results table in `REPORT.md`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Research relationship

This is a transparent coursework implementation inspired by MultiLoRA, MeteoRA, and MALoRA. It demonstrates low-rank task experts, sparse expert routing, multi-task training, and quantitative routing comparisons. It is not an exact reproduction of any one paper.
