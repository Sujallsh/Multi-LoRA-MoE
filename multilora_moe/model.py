from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class ModelConfig:
    model_name: str
    task_num_labels: dict[str, int]
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    top_k: int = 2


class LoRAFeatureExpert(nn.Module):
    """Low-rank residual adapter operating on pooled encoder features."""

    def __init__(self, hidden_size: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.a = nn.Linear(hidden_size, rank, bias=False)
        self.b = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.b(self.a(self.dropout(hidden))) * self.scale


class ExpertRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, num_experts),
        )
        self.top_k = min(max(1, top_k), num_experts)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(hidden)
        probabilities = torch.softmax(logits, dim=-1)
        if self.top_k < probabilities.shape[-1]:
            values, indices = probabilities.topk(self.top_k, dim=-1)
            sparse = torch.zeros_like(probabilities).scatter(-1, indices, values)
            probabilities = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return logits, probabilities


class MultiLoRAMoE(nn.Module):
    def __init__(self, model_name, task_num_labels, rank=8, alpha=16.0, dropout=0.1, top_k=2):
        super().__init__()
        self.model_config = ModelConfig(model_name, task_num_labels, rank, alpha, dropout, top_k)
        self.task_names = list(task_num_labels)
        self.task_to_index = {name: index for index, name in enumerate(self.task_names)}
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.experts = nn.ModuleList(
            LoRAFeatureExpert(hidden_size, rank, alpha, dropout) for _ in self.task_names
        )
        self.router = ExpertRouter(hidden_size, len(self.experts), top_k)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_size, labels) for name, labels in task_num_labels.items()}
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _masked_mean(hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        with torch.no_grad():
            output = self.encoder(**encoder_inputs)
        return self._masked_mean(output.last_hidden_state, attention_mask)

    def forward(
        self,
        input_ids,
        attention_mask,
        task_name: str,
        routing: str = "moe",
        token_type_ids=None,
    ):
        pooled = self.encode(input_ids, attention_mask, token_type_ids)
        expert_features = torch.stack([expert(pooled) for expert in self.experts], dim=1)
        router_logits, router_weights = self.router(pooled)
        if routing == "task_expert":
            router_weights = torch.zeros_like(router_weights)
            router_weights[:, self.task_to_index[task_name]] = 1.0
        elif routing == "uniform":
            router_weights = torch.full_like(router_weights, 1.0 / len(self.experts))
        elif routing != "moe":
            raise ValueError("routing must be 'moe', 'task_expert', or 'uniform'")
        combined = torch.sum(expert_features * router_weights.unsqueeze(-1), dim=1)
        return {
            "logits": self.heads[task_name](self.dropout(combined)),
            "router_logits": router_logits,
            "router_weights": router_weights,
            "features": combined,
        }

    def set_stage(self, stage: str, task_name: str | None = None):
        for parameter in self.parameters():
            parameter.requires_grad = False
        if stage == "expert":
            if task_name is None:
                raise ValueError("task_name is required for the expert stage")
            for parameter in self.experts[self.task_to_index[task_name]].parameters():
                parameter.requires_grad = True
            for parameter in self.heads[task_name].parameters():
                parameter.requires_grad = True
        elif stage == "router":
            for parameter in self.router.parameters():
                parameter.requires_grad = True
            for head in self.heads.values():
                for parameter in head.parameters():
                    parameter.requires_grad = True
        elif stage == "joint":
            for expert in self.experts:
                for parameter in expert.parameters():
                    parameter.requires_grad = True
            for parameter in self.router.parameters():
                parameter.requires_grad = True
            for head in self.heads.values():
                for parameter in head.parameters():
                    parameter.requires_grad = True
        else:
            raise ValueError("stage must be 'expert', 'router', or 'joint'")

    def checkpoint(self):
        return {"config": asdict(self.model_config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint, map_location="cpu"):
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(map_location)
