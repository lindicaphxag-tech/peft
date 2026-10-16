"""Regression coverage for native PyTorch TransformerEncoder fast paths."""

import copy

import torch
from torch import nn

from peft import LoraConfig, get_peft_model


def test_unmerged_lora_is_not_bypassed_by_transformer_encoder_fastpath():
    torch.manual_seed(42)

    layer = nn.TransformerEncoderLayer(
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        dropout=0,
        batch_first=True,
    )
    model = get_peft_model(
        nn.TransformerEncoder(layer, num_layers=2),
        LoraConfig(r=4, lora_alpha=8, target_modules=["linear1", "linear2"]),
    )
    base = copy.deepcopy(model.base_model.model)

    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "lora_B") and "default" in module.lora_B:
                module.lora_B["default"].weight.fill_(0.1)

    x = torch.randn(2, 6, 16)
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        model.eval()
        with torch.inference_mode():
            torch.backends.mha.set_fastpath_enabled(True)
            fastpath_output = model(x)

            torch.backends.mha.set_fastpath_enabled(False)
            reference_output = model(x)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)

    assert not torch.equal(fastpath_output, base(x))
    torch.testing.assert_close(fastpath_output, reference_output)
