"""Low-precision LoRA tied adapter storage and optimizer invariants.

Reproduces upstream issue #3929 without a model download.
"""
import pytest
import torch

from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


def tiny_model(dtype, autocast_adapter_dtype):
    torch.manual_seed(0)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    ).to(dtype)
    cfg = LoraConfig(
        target_modules=["embed_tokens", "lm_head"],
        ensure_weight_tying=True, init_lora_weights=False,
    )
    return get_peft_model(base, cfg, autocast_adapter_dtype=autocast_adapter_dtype)


def tied_parts(model, name="default"):
    base = model.base_model.model
    emb = base.model.embed_tokens
    head = base.lm_head
    return (emb.lora_embedding_A[name], emb.lora_embedding_B[name],
            head.lora_A[name].weight, head.lora_B[name].weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("autocast_adapter_dtype", [False, True])
def test_tied_lora_storage_across_dtype_cast_and_training_step(dtype, autocast_adapter_dtype):
    model = tiny_model(dtype, autocast_adapter_dtype)
    emb_a, emb_b, head_a, head_b = tied_parts(model)
    assert emb_b.data_ptr() == head_a.data_ptr()
    assert emb_a.data_ptr() == head_b.data_ptr()
    assert torch.equal(head_a, emb_b.T)
    assert torch.equal(head_b, emb_a.T)
    wanted = torch.float32 if autocast_adapter_dtype else dtype
    assert head_a.dtype == head_b.dtype == emb_a.dtype == emb_b.dtype == wanted

    # An optimizer must see the same storage relationship *after* the update.
    # This is the contract violated by the old generic independent upcast.
    # An fp16 CPU forward/backward may use unsupported kernels; optimize a
    # differentiable LoRA-parameter-only objective to isolate tied storage.
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    opt.zero_grad()
    loss = (emb_a.float().square().mean() + emb_b.float().square().mean()
            + head_a.float().square().mean() + head_b.float().square().mean())
    loss.backward()
    opt.step()
    emb_a, emb_b, head_a, head_b = tied_parts(model)
    assert emb_b.data_ptr() == head_a.data_ptr()
    assert emb_a.data_ptr() == head_b.data_ptr()
    torch.testing.assert_close(head_a, emb_b.T, rtol=0, atol=0)
    torch.testing.assert_close(head_b, emb_a.T, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tied_lora_real_forward_after_upcast(dtype):
    model = tiny_model(dtype, True)
    assert model.base_model.model.lm_head.lora_A["default"].weight.dtype == torch.float32
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert torch.isfinite(outputs.logits).all()
