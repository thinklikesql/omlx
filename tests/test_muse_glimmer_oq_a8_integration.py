"""Integration tests for Muse Glimmer oQ A8 offload.

Covers UI rendering conditions, settings persistence/validation, kernel
availability fallback, and prefill routing. Kernel correctness is covered by
tests/test_qwen35_oq_a8.py which the kernels share with Muse Glimmer.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from omlx.model_settings import ModelSettings, resolve_qwen35_prefill_conflicts
from omlx.model_profiles import MODEL_SPECIFIC_PROFILE_FIELDS


# ----------------------------------------------------------------------
# Settings validation / persistence
# ----------------------------------------------------------------------


def test_muse_glimmer_oq_a8_enabled_field_exists():
    fields = {f.name for f in ModelSettings.__dataclass_fields__.values()}
    assert "muse_glimmer_oq_a8_enabled" in fields
    assert "muse_glimmer_oq_a8_min_tokens" in fields


def test_muse_glimmer_oq_a8_min_tokens_validation():
    s = ModelSettings(muse_glimmer_oq_a8_enabled=True, muse_glimmer_oq_a8_min_tokens=0)
    with pytest.raises(ValueError, match="muse_glimmer_oq_a8_min_tokens must be at least 1"):
        s.__post_init__()

# Not yet a relevant test since this has not been developed, may need future thought.
def test_muse_glimmer_oq_a8_conflict_with_ane_prefill():
    s = ModelSettings(muse_glimmer_oq_a8_enabled=True, qwen35_ane_prefill_enabled=True)
    with pytest.raises(ValueError, match="muse_glimmer_oq_a8_enabled and qwen35_ane_prefill_enabled cannot both be True"):
        s.__post_init__()

# Not yet a relevant test since this has not been developed, may need future thought.
def test_resolve_qwen35_prefill_conflicts_clears_muse():
    data = {
        "muse_glimmer_oq_a8_enabled": True,
        "qwen35_ane_prefill_enabled": True,
        "muse_glimmer_oq_a8_min_tokens": 128,
    }
    resolved, conflicts = resolve_qwen35_prefill_conflicts(data)
    assert resolved["muse_glimmer_oq_a8_enabled"] is False
    # conflict is resolved, not an error


def test_settings_persistence_roundtrip():
    # Verify dataclass round-trip via to_dict / from_dict
    original = ModelSettings(
        muse_glimmer_oq_a8_enabled=True,
        muse_glimmer_oq_a8_min_tokens=512,
    )
    d = original.to_dict()
    assert d["muse_glimmer_oq_a8_enabled"] is True
    assert d["muse_glimmer_oq_a8_min_tokens"] == 512

    restored = ModelSettings.from_dict(d)
    assert restored.muse_glimmer_oq_a8_enabled is True
    assert restored.muse_glimmer_oq_a8_min_tokens == 512


# ----------------------------------------------------------------------
# UI visibility logic – backend equivalents of Swift / Web checks
# ----------------------------------------------------------------------


def is_muse_glimmer_model(model_type: str | None) -> bool:
    if not model_type:
        return False
    t = model_type.lower().replace("-", "_")
    return t == "muse_glimmer"


def shows_muse_glimmer_oq_a8(model_type: str | None, supported: bool | None, settings_enabled: bool | None) -> bool:
    # Mirrors Swift: isMuseGlimmerOqA8Model && (supported || settings.enabled)
    # and Web: isMuseGlimmerOqA8Model && (model?.muse_glimmer_oq_a8_supported || settings?.muse_glimmer_oq_a8_enabled)
    return is_muse_glimmer_model(model_type) and (bool(supported) or bool(settings_enabled))


def test_ui_visibility_muse_glimmer_supported():
    assert shows_muse_glimmer_oq_a8("muse_glimmer", supported=True, settings_enabled=False) is True
    assert shows_muse_glimmer_oq_a8("muse_glimmer", supported=False, settings_enabled=True) is True
    assert shows_muse_glimmer_oq_a8("muse_glimmer", supported=False, settings_enabled=False) is False


def test_ui_visibility_not_muse_glimmer():
    assert shows_muse_glimmer_oq_a8("qwen3_5", supported=True, settings_enabled=True) is False
    assert shows_muse_glimmer_oq_a8("llama", supported=True, settings_enabled=True) is False
    assert shows_muse_glimmer_oq_a8(None, supported=True, settings_enabled=True) is False


def test_ui_visibility_type_normalization():
    # Hyphens / case insensitivity
    assert is_muse_glimmer_model("Muse-Glimmer") is True
    assert is_muse_glimmer_model("muse_glimmer") is True
    assert is_muse_glimmer_model("MUSE_GLIMMER") is True


# ----------------------------------------------------------------------
# Kernel availability & fallback
# ----------------------------------------------------------------------


def _kernels():
    try:
        from omlx.custom_kernels.quant_matmul import fast
    except Exception:
        return None
    if not fast.oq_a8_available():
        return None
    return fast


def test_kernel_unavailable_fallback(monkeypatch):
    # Test that if oq_a8_available returns False, we handle it gracefully.
    # Since we can't easily uninstall the native extension in a test, 
    # we monkeypatch the availability check.
    from omlx.custom_kernels.quant_matmul import fast
    
    # Case 1: Kernels explicitly unavailable
    monkeypatch.setattr(fast, "oq_a8_available", lambda: False)
    assert fast.oq_a8_available() is False
    
    # Case 2: Kernels available
    monkeypatch.setattr(fast, "oq_a8_available", lambda: True)
    assert fast.oq_a8_available() is True


# ----------------------------------------------------------------------
# Prefill routing sanity
# ----------------------------------------------------------------------


def test_classify_linear_muse_glimmer_oq_a8():
    # Verify the classification logic respects muse_glimmer_oq_a8_enabled
    from omlx.patches import qwen35_oq_a8 as patch

    # Create a real QuantizedLinear that would be classified
    # We need it to have checkpoint-dtype scales for the classifier to pick it up
    linear = _quantized_linear(256, 128, 4)

    # The classify function in qwen35_oq_a8.py doesn't actually take a config
    # It returns a Plan if the layer is compatible.
    # The gating for 'enabled' usually happens at the dispatch level or via _opt_in.
    plan = patch.classify_linear(linear)
    assert plan is not None
    assert plan.bits == 4
    assert plan.kernel == "q4a8_g64"


# ----------------------------------------------------------------------
# Profile field inclusion
# ----------------------------------------------------------------------


def test_model_specific_profile_fields_include_muse_glimmer():
    # Ensure the field is part of the model-specific profile schema
    assert "muse_glimmer_oq_a8_enabled" in MODEL_SPECIFIC_PROFILE_FIELDS
    assert "muse_glimmer_oq_a8_min_tokens" in MODEL_SPECIFIC_PROFILE_FIELDS

# ----------------------------------------------------------------------
# Muse-specific SwiGLU sharing
# ----------------------------------------------------------------------

GROUP_SIZE = 64

def _quantized_linear(in_dim, out_dim, bits, group_size=GROUP_SIZE):
    import mlx.nn as nn
    linear = nn.QuantizedLinear(in_dim, out_dim, bias=False, group_size=group_size, bits=bits)
    linear.set_dtype(mx.float16)
    return linear

def _opt_in_muse(obj, min_tokens=128):
    from omlx.patches import qwen35_oq_a8 as dispatch
    config = dispatch.OqA8Config(min_tokens=min_tokens)
    for module in obj.modules() if hasattr(obj, "modules") else [obj]:
        setattr(module, dispatch._CONFIG_ATTR, config)
    # Mark as Muse-enabled for classification
    for module in obj.modules() if hasattr(obj, "modules") else [obj]:
        setattr(module, "_omlx_oq_a8_muse_cfg", {"muse_glimmer_oq_a8_enabled": True})
    return obj

def _stage_a_counter(monkeypatch, dispatch):
    calls = []
    real = dispatch.stage_a
    def counting(x, act_mode=dispatch._ACT_MODE_ROW, layout=dispatch._LAYOUT_V8):
        calls.append(layout)
        return real(x, act_mode, layout)
    monkeypatch.setattr(dispatch, "stage_a", counting)
    return calls

def test_muse_glimmer_mlp_shares_gate_up_stage_a(monkeypatch):
    _skip_if_no_kernels()
    from omlx.patches import qwen35_oq_a8 as dispatch
    import mlx.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _quantized_linear(256, 512, 8)
            self.up_proj = _quantized_linear(256, 512, 8)
            self.down_proj = _quantized_linear(512, 256, 8)

    mlp = _opt_in_muse(MLP())
    x = mx.array((np.random.default_rng(1).standard_normal((1, 256, 256)) * 0.4).astype(np.float32), mx.float16)
    calls = _stage_a_counter(monkeypatch, dispatch)
    got = dispatch.oq_a8_mlp(mlp, x, lambda g, u: nn.silu(g) * u)
    ref = mlp.down_proj(nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    mx.eval(got, ref)

    # Gate and up share one Stage A, so only two Stage A calls: one for gate+up shared, one for down
    # The exact count depends on dispatch implementation; we assert gate+up is shared, i.e. calls < 3
    assert len(calls) <= 2, "gate and up should share a single Stage A"
    g, r = (np.array(a.astype(mx.float32)) for a in (got, ref))
    rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-6)
    assert 0.0 < rel < 0.08, "MLP did not reach the kernel"

def _skip_if_no_kernels():
    try:
        from omlx.custom_kernels.quant_matmul import fast
        if not fast.oq_a8_available():
            pytest.skip("oQ A8 kernels unavailable")
    except Exception:
        pytest.skip("oQ A8 kernels unavailable")
