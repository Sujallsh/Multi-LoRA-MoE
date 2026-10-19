from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    num_labels: int
    text_builder: Callable[[dict], str]
    label_names: tuple[str, ...]
    train_split: str = "train"
    eval_split: str = "test"


TASKS: dict[str, TaskSpec] = {
    "ag_news": TaskSpec(
        "fancyzhx/ag_news",
        4,
        lambda row: row["text"],
        ("World", "Sports", "Business", "Science and Technology"),
    ),
    "emotion": TaskSpec(
        "dair-ai/emotion",
        6,
        lambda row: row["text"],
        ("sadness", "joy", "love", "anger", "fear", "surprise"),
        eval_split="validation",
    ),
    "dbpedia": TaskSpec(
        "fancyzhx/dbpedia_14",
        14,
        lambda row: f"{row.get('title', '')} {row.get('content', '')}".strip(),
        (
            "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
            "Means of Transportation", "Building", "Natural Place", "Village", "Animal",
            "Plant", "Album", "Film", "Written Work",
        ),
    ),
}


def _sample(dataset, limit: int, seed: int):
    shuffled = dataset.shuffle(seed=seed)
    if limit <= 0 or limit >= len(shuffled):
        return shuffled
    return shuffled.select(range(limit))


def load_task(task_name, tokenizer, train_samples, eval_samples, max_length, seed):
    from datasets import load_dataset

    spec = TASKS[task_name]
    train = load_dataset(spec.dataset, split=spec.train_split)
    evaluation = load_dataset(spec.dataset, split=spec.eval_split)
    train = _sample(train, train_samples, seed)
    evaluation = _sample(evaluation, eval_samples, seed + 1)

    def encode(batch):
        rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        encoded = tokenizer(
            [spec.text_builder(row) for row in rows], truncation=True, max_length=max_length
        )
        encoded["labels"] = batch["label"]
        return encoded

    train = train.map(encode, batched=True, remove_columns=train.column_names)
    evaluation = evaluation.map(encode, batched=True, remove_columns=evaluation.column_names)
    train.set_format("torch")
    evaluation.set_format("torch")
    return train, evaluation


def make_loader(dataset, tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
    )
