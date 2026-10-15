"""OFT Linear must match sequential merge for non-commuting adapters.

The regression deliberately uses two or three non-identity rotations; a
single adapter cannot expose the order reversal.
"""
import copy

import pytest
import torch
from torch import nn

from peft import OFTConfig, get_peft_model


class _LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 4)

    def forward(self, x):
        return self.proj(x)


@pytest.mark.parametrize("num_adapters", [2, 3])
@pytest.mark.parametrize("coft", [False, True])
def test_linear_multiple_oft_adapters_match_merge(num_adapters, coft):
    torch.manual_seed(12345)
    x = torch.randn(11, 6)
    cfg = OFTConfig(r=2, oft_block_size=0, target_modules=["proj"], coft=coft)
    model = get_peft_model(_LinearModel(), cfg).eval()
    names = ["default"]
    for idx in range(1, num_adapters):
        name = f"additional_{idx}"
        model.add_adapter(name, copy.deepcopy(cfg))
        names.append(name)
    model.base_model.set_adapter(names)

    # Force finite nontrivial skew parameters; identity rotations commute.
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "oft_R" in name and "weight" in name:
                value.copy_(torch.randn_like(value) * 0.10)

    reference = model(x).detach().clone()
    model.merge_adapter()
    merged = model(x).detach().clone()
    torch.testing.assert_close(reference, merged, rtol=3e-5, atol=3e-5)
    model.unmerge_adapter()
    restored = model(x).detach().clone()
    torch.testing.assert_close(reference, restored, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("coft", [False, True])
def test_multi_adapter_oft_gradients_reach_each_rotation(coft):
    torch.manual_seed(123)
    cfg = OFTConfig(r=2, oft_block_size=0, target_modules=["proj"], coft=coft)
    model = get_peft_model(_LinearModel(), cfg).train()
    model.add_adapter("other", copy.deepcopy(cfg))
    model.base_model.set_adapter(["default", "other"])
    model(torch.randn(10, 6)).square().mean().backward()
    trainable = [(name, p) for name, p in model.named_parameters()
                 if "oft_R" in name and p.requires_grad]
    assert len(trainable) >= 2
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for _, p in trainable)
    assert all(p.grad.abs().max() > 0 for _, p in trainable)
