import torch
from torch import nn

from multilora_moe.model import ExpertRouter, LoRAFeatureExpert


def test_lora_starts_as_identity():
    expert = LoRAFeatureExpert(hidden_size=12, rank=3, alpha=6, dropout=0)
    hidden = torch.randn(4, 12)
    assert torch.allclose(expert(hidden), hidden)


def test_router_weights_sum_to_one():
    router = ExpertRouter(hidden_size=12, num_experts=3, top_k=2)
    _, weights = router(torch.randn(5, 12))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5))
    assert torch.all((weights > 0).sum(dim=-1) <= 2)


def test_lora_receives_gradients():
    expert = LoRAFeatureExpert(hidden_size=8, rank=2, alpha=4, dropout=0)
    loss = nn.functional.mse_loss(expert(torch.randn(3, 8)), torch.zeros(3, 8))
    loss.backward()
    assert expert.b.weight.grad is not None

