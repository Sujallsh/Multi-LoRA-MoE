from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm import tqdm


def trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_expert_epoch(model, loader, task_name, optimizer, device):
    model.train()
    losses = []
    for batch in tqdm(loader, desc=f"expert:{task_name}", leave=False):
        batch = _move(batch, device)
        labels = batch.pop("labels")
        output = model(**batch, task_name=task_name, routing="task_expert")
        loss = nn.functional.cross_entropy(output["logits"], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def train_mixed_epoch(model, loaders, optimizer, device, router_loss_weight):
    model.train()
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    active = list(iterators)
    losses = []
    while active:
        task_name = random.choice(active)
        try:
            batch = next(iterators[task_name])
        except StopIteration:
            active.remove(task_name)
            continue
        batch = _move(batch, device)
        labels = batch.pop("labels")
        output = model(**batch, task_name=task_name, routing="moe")
        classification_loss = nn.functional.cross_entropy(output["logits"], labels)
        router_targets = torch.full(
            (labels.shape[0],), model.task_to_index[task_name], device=device, dtype=torch.long
        )
        router_loss = nn.functional.cross_entropy(output["router_logits"], router_targets)
        loss = classification_loss + router_loss_weight * router_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def evaluate(model, loaders, device, routing="moe"):
    model.eval()
    results = {}
    router_totals = torch.zeros(len(model.task_names), device=device)
    router_count = 0
    for task_name, loader in loaders.items():
        predictions, references = [], []
        for batch in loader:
            batch = _move(batch, device)
            labels = batch.pop("labels")
            output = model(**batch, task_name=task_name, routing=routing)
            predictions.extend(output["logits"].argmax(dim=-1).cpu().tolist())
            references.extend(labels.cpu().tolist())
            router_totals += output["router_weights"].sum(dim=0)
            router_count += labels.shape[0]
        results[task_name] = {
            "accuracy": accuracy_score(references, predictions),
            "macro_f1": f1_score(references, predictions, average="macro", zero_division=0),
            "samples": len(references),
        }
    results["average_macro_f1"] = sum(
        results[name]["macro_f1"] for name in loaders
    ) / len(loaders)
    results["router_usage"] = {
        name: float(router_totals[index].item() / max(1, router_count))
        for index, name in enumerate(model.task_names)
    }
    return results


def save_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

