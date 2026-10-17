import copy

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model


class NativeTransformerEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_gradient_checkpointing = False
        layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


@pytest.mark.parametrize("use_padding_mask", [False, True])
def test_lora_adapter_is_preserved_with_transformer_encoder_fastpath(use_padding_mask):
    if not hasattr(torch.backends, "mha") or not all(
        hasattr(torch.backends.mha, name) for name in ("get_fastpath_enabled", "set_fastpath_enabled")
    ):
        pytest.skip("PyTorch TransformerEncoder fastpath controls are unavailable")

    torch.manual_seed(42)
    base = NativeTransformerEncoderModel()
    reference = copy.deepcopy(base).eval()
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["linear1", "linear2"],
            init_lora_weights=False,
        ),
    ).eval()

    x = torch.randn(2, 6, 16)
    padding_mask = None
    if use_padding_mask:
        # Left-aligned valid tokens exercise TransformerEncoder's nested-tensor path.
        padding_mask = torch.tensor(
            [
                [False, False, False, True, True, True],
                [False, False, False, False, True, True],
            ]
        )

    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        with torch.inference_mode():
            torch.backends.mha.set_fastpath_enabled(True)
            fast_output = model(x, src_key_padding_mask=padding_mask)

            torch.backends.mha.set_fastpath_enabled(False)
            slow_output = model(x, src_key_padding_mask=padding_mask)
            base_output = reference(x, src_key_padding_mask=padding_mask)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)

    # Adapter-aware execution must not depend on PyTorch's fused-path setting.
    torch.testing.assert_close(fast_output, slow_output, rtol=1e-5, atol=1e-6)
    # The adapter is initialized non-zero; its contribution must not disappear at inference.
    assert not torch.allclose(slow_output, base_output, rtol=1e-5, atol=1e-6)
