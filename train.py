from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from multilora_moe.data import TASKS, load_task, make_loader
from multilora_moe.model import MultiLoRAMoE
from multilora_moe.training import (
    evaluate,
    save_json,
    train_expert_epoch,
    train_mixed_epoch,
    trainable_parameters,
)


def arguments():
    parser = argparse.ArgumentParser(description="Train Multi-LoRA experts and an MoE router")
    parser.add_argument("--model-name", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--tasks", nargs="+", default=["ag_news", "emotion", "dbpedia"])
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-epochs", type=int, default=1)
    parser.add_argument("--router-epochs", type=int, default=2)
    parser.add_argument("--joint-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--router-loss-weight", type=float, default=0.25)
    parser.add_argument("--output-dir", default="outputs/cpu_demo")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    unknown = set(args.tasks) - set(TASKS)
    if unknown:
        raise SystemExit(f"Unknown tasks: {sorted(unknown)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    task_num_labels = {name: TASKS[name].num_labels for name in args.tasks}

    train_loaders, eval_loaders = {}, {}
    print("Loading and tokenizing datasets...")
    for task_name in args.tasks:
        train_data, eval_data = load_task(
            task_name, tokenizer, args.train_samples, args.eval_samples, args.max_length, args.seed
        )
        train_loaders[task_name] = make_loader(train_data, tokenizer, args.batch_size, True)
        eval_loaders[task_name] = make_loader(eval_data, tokenizer, args.batch_size, False)

    model = MultiLoRAMoE(
        args.model_name,
        task_num_labels,
        rank=args.rank,
        alpha=args.alpha,
        top_k=args.top_k,
    ).to(device)
    history = {"device": str(device), "config": vars(args), "stages": []}

    for task_name in args.tasks:
        model.set_stage("expert", task_name)
        trainable, total = trainable_parameters(model)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
        )
        for epoch in range(args.expert_epochs):
            loss = train_expert_epoch(model, train_loaders[task_name], task_name, optimizer, device)
            record = {
                "stage": "expert", "task": task_name, "epoch": epoch + 1, "loss": loss,
                "trainable_parameters": trainable, "total_parameters": total,
            }
            history["stages"].append(record)
            print(json.dumps(record))

    model.set_stage("router")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )
    for epoch in range(args.router_epochs):
        loss = train_mixed_epoch(
            model, train_loaders, optimizer, device, args.router_loss_weight
        )
        metrics = evaluate(model, eval_loaders, device)
        record = {"stage": "router", "epoch": epoch + 1, "loss": loss, "metrics": metrics}
        history["stages"].append(record)
        print(json.dumps(record))

    if args.joint_epochs:
        model.set_stage("joint")
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate / 5
        )
        for epoch in range(args.joint_epochs):
            loss = train_mixed_epoch(
                model, train_loaders, optimizer, device, args.router_loss_weight
            )
            metrics = evaluate(model, eval_loaders, device)
            record = {"stage": "joint", "epoch": epoch + 1, "loss": loss, "metrics": metrics}
            history["stages"].append(record)
            print(json.dumps(record))

    comparison = {
        mode: evaluate(model, eval_loaders, device, routing=mode)
        for mode in ("uniform", "moe", "task_expert")
    }
    history["comparison"] = comparison
    torch.save(model.checkpoint(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    save_json(history, output_dir / "training_history.json")
    save_json(comparison, output_dir / "comparison.json")
    print(f"Saved model and results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
