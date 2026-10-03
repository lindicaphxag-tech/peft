import pytest
from torch import nn

from peft import LoraConfig, get_peft_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


def make_config():
    return LoraConfig(r=2, lora_alpha=2, target_modules=["proj"], init_lora_weights=False)


# Regression test for huggingface/peft#3872.
def test_set_adapter_unmerges_before_fsdp_reshard(monkeypatch):
    model = get_peft_model(TinyModel(), make_config())
    model.add_adapter("other", make_config())
    model.merge_adapter()

    events = []
    original_unmerge = model.base_model.unmerge_adapter
    original_set_adapter = model.base_model.set_adapter

    def unmerge_spy():
        events.append("unmerge")
        return original_unmerge()

    def reshard_spy():
        events.append("reshard")

    def set_adapter_spy(adapter_name, inference_mode=False):
        events.append("set_adapter")
        return original_set_adapter(adapter_name, inference_mode=inference_mode)

    monkeypatch.setattr(model.base_model, "unmerge_adapter", unmerge_spy)
    monkeypatch.setattr(model, "_reshard_fsdp_modules", reshard_spy)
    monkeypatch.setattr(model.base_model, "set_adapter", set_adapter_spy)

    with pytest.warns(UserWarning, match="Unmerging the model first"):
        model.set_adapter("other")

    assert events == ["unmerge", "reshard", "set_adapter"]
    assert model.active_adapter == "other"


def test_set_adapter_does_not_unmerge_when_model_is_unmerged(monkeypatch):
    model = get_peft_model(TinyModel(), make_config())
    model.add_adapter("other", make_config())

    events = []
    original_set_adapter = model.base_model.set_adapter

    def unexpected_unmerge():
        raise AssertionError("unmerge_adapter() should not run for an unmerged model")

    def reshard_spy():
        events.append("reshard")

    def set_adapter_spy(adapter_name, inference_mode=False):
        events.append("set_adapter")
        return original_set_adapter(adapter_name, inference_mode=inference_mode)

    monkeypatch.setattr(model.base_model, "unmerge_adapter", unexpected_unmerge)
    monkeypatch.setattr(model, "_reshard_fsdp_modules", reshard_spy)
    monkeypatch.setattr(model.base_model, "set_adapter", set_adapter_spy)

    model.set_adapter("other")

    assert events == ["reshard", "set_adapter"]
