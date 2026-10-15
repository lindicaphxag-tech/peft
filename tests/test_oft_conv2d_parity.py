"""Regression for OFT Conv2d weight-space rotations (issue #3829).

Overlapping input patches must not be folded back before convolution: the
unmerged adapter should implement exactly the same weight as merge().
"""
import copy

import pytest
import torch
from torch import nn

from peft import OFTConfig, get_peft_model


class _ConvModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(4, 6, **kwargs)

    def forward(self, x):
        return self.conv(x)


CONVOLUTIONS = [
    {"kernel_size": 1},
    {"kernel_size": 3, "padding": 1},
    {"kernel_size": (3, 5), "padding": (1, 2)},
    {"kernel_size": 3, "padding": 1, "groups": 2},
    {"kernel_size": 3, "padding": 1, "padding_mode": "circular"},
    {"kernel_size": (3, 5), "padding": (1, 2), "stride": (1, 2)},
]


@pytest.mark.parametrize("conv_args", CONVOLUTIONS)
@pytest.mark.parametrize("coft", [False, True])
@pytest.mark.parametrize("init_weights", [False, True])
def test_oft_conv_forward_matches_merge_and_unmerge(conv_args, coft, init_weights):
    torch.manual_seed(42)
    x = torch.randn(2, 4, 9, 11)
    base = _ConvModel(**conv_args).eval()
    initial = base(x).detach()
    peft_model = get_peft_model(
        copy.deepcopy(base),
        OFTConfig(
            r=2, oft_block_size=0, target_modules=["conv"],
            init_weights=init_weights, coft=coft,
        ),
    ).eval()
    before = peft_model(x).detach()
    if init_weights:
        torch.testing.assert_close(before, initial, rtol=1e-5, atol=1e-5)
    peft_model.merge_adapter()
    after_merge = peft_model(x).detach()
    torch.testing.assert_close(before, after_merge, rtol=2e-5, atol=2e-5)
    peft_model.unmerge_adapter()
    after_unmerge = peft_model(x).detach()
    torch.testing.assert_close(before, after_unmerge, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("coft", [False, True])
def test_oft_conv_training_gradient_reaches_rotation_parameter(coft):
    torch.manual_seed(1234)
    base = _ConvModel(kernel_size=3, padding=1)
    model = get_peft_model(
        base,
        OFTConfig(r=2, oft_block_size=0, target_modules=["conv"], coft=coft),
    ).train()
    x = torch.randn(2, 4, 7, 8)
    model(x).square().mean().backward()
    trainable_rotations = [
        param for name, param in model.named_parameters()
        if "oft_R" in name and param.requires_grad
    ]
    assert trainable_rotations
    assert all(param.grad is not None and torch.isfinite(param.grad).all()
               for param in trainable_rotations)
    assert any(param.grad.abs().max() > 0 for param in trainable_rotations)
    # The frozen base convolution must not receive adapter-training gradients.
    assert all(
        param.grad is None for name, param in model.named_parameters()
        if "base_layer.weight" in name and not param.requires_grad
    )
