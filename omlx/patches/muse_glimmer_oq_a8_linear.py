"""First-refusal linear wrapper for Muse Glimmer oQ A8 prefill.

Mirrors gemma4_oq_a8_linear.py but for the Muse Glimmer model family. Muse
Glimmer's attention is standard MHA (q/k/v/o_proj) with an additional
gate_proj that modulates the attention output. The shared oQ A8 kernels
(NAX int8xint8->int32 with a FP32 affine correction) are model-agnostic, so
this is a pure surface change: reclassify the attention QuantizedLinears into
a first-refusal wrapper that gives the int8-A8 backend (_prefill_linear_backend
reused from qwen35_oq_a8) a chance; if it declines (not tagged, not
Q8/affine-eligible, decode-shaped, etc.) it falls back to stock
nn.QuantizedLinear.__call__.

The dispatch module (muse_glimmer_oq_a8.py) imports this one lazily inside
its apply function; this module imports no omlx code, so each stays
importable in isolation.
"""

from __future__ import annotations

import logging

import mlx.nn as nn

logger = logging.getLogger(__name__)

# Module-global backend set by register_muse_glimmer_prefill_linear_backend
# (called from muse_glimmer_oq_a8.apply_muse_glimmer_oq_a8_patch). Mirrors
# _PREFILL_LINEAR_BACKEND in qwen35_q4_mlp.py.
_PREFILL_LINEAR_BACKEND = None

# Projection names Muse Glimmer exposes on its Attention. MLP gate/up/down are
# handled by the direct MLP-class patch in muse_glimmer_oq_a8 (shared Stage A);
# the linear wrapper here is for the attention projections only. Muse Glimmer's
# attention has an extra gate_proj that Gemma-4's does not.
_PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj")

# Module-path prefix whose QuantizedLinear projections may be reclassified.
# Muse Glimmer has two model namespaces depending on the engine path:
# - mlx_vlm.models.muse_glimmer.* — VLMBatchedEngine (vlm path)
# - mlx_lm.models.muse_glimmer.* — BatchedEngine (force_lm=True, benchmark path)
# Both must be matched so attention projections are reclassified to the
# int8-A8 first-refusal wrapper regardless of which loader produced the model.
_MUSE_GLIMMER_MODULE_PREFIXES = (
    "mlx_vlm.models.muse_glimmer.",
    "mlx_lm.models.muse_glimmer.",
)


def _module_matches(module: str) -> bool:
    """True when ``module`` is one of the Muse Glimmer model modules or
    lives inside one of its packages."""
    for entry in _MUSE_GLIMMER_MODULE_PREFIXES:
        root = entry.rstrip(".")
        if module == root or module.startswith(root + "."):
            return True
    return False


def register_muse_glimmer_prefill_linear_backend(backend) -> None:
    """Register the int8-A8 first-refusal backend (mirrors
    register_gemma4_prefill_linear_backend)."""
    global _PREFILL_LINEAR_BACKEND
    _PREFILL_LINEAR_BACKEND = backend


def _try_backend(linear, x):
    backend = _PREFILL_LINEAR_BACKEND
    if backend is None:
        return None
    try:
        return backend(linear, x)
    except Exception:
        logger.debug(
            "muse_glimmer prefill linear backend failed; falling back",
            exc_info=True,
        )
        return None


class _MuseGlimmerQuantizedPrefillLinear(nn.QuantizedLinear):
    """Attention projection that first-refuses to the int8-A8 backend.

    Mirrors _GemmaQuantizedPrefillLinear: a Muse Glimmer linear either goes
    int8-A8 (if tagged+eligible) or stock. The backend
    (muse_glimmer_oq_a8._prefill_linear_backend, reused from
    qwen35_oq_a8) returns None on decline so the wrapper falls back to stock
    __call__ without re-entering itself.
    """

    def __call__(self, x):
        routed = _try_backend(self, x)
        if routed is not None:
            return routed
        return super().__call__(x)


def apply_muse_glimmer_oq_a8_prefill_linear_patch(model) -> bool:
    """Reclassify Muse Glimmer attention QuantizedLinears into the
    first-refusal wrapper. Mirrors apply_gemma4_oq_a8_prefill_linear_patch
    (the module-path filter + per-name reclassification), but restricted to
    q/k/v/o/gate_proj and QuantizedLinear only (no PackedLinear).

    MLP instances are skipped even though they match the module prefix and
    also expose a gate_proj: their gate/up/down are the shared-Stage-A
    surface installed by the direct MLP-class patch in muse_glimmer_oq_a8,
    so the only gate_proj reclassified here is the attention one.

    Must run on every model load: the reclassification is per-instance
    (``linear.__class__ = ...``), so callers must not hide it behind a
    process-wide one-shot guard. Already-wrapped linears fail the exact-type
    check, which makes repeat calls idempotent.
    """
    installed = False
    for _, module in model.named_modules():
        mod = type(module).__module__ or ""
        if not _module_matches(mod):
            continue
        # MLP: gate/up/down are the shared-Stage-A surface patched directly
        # on the MLP class; reclassifying its gate_proj here would leave it
        # on a different path than up/down in the MLP's fallback forward.
        if hasattr(module, "up_proj") and hasattr(module, "down_proj"):
            continue
        for name in _PROJECTION_NAMES:
            linear = getattr(module, name, None)
            if type(linear) is nn.QuantizedLinear:
                linear.__class__ = _MuseGlimmerQuantizedPrefillLinear
                installed = True
    if installed:
        logger.info("muse_glimmer oQ A8 prefill linear wrappers installed")
    return installed
