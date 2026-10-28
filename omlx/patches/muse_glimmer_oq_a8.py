# ruff: noqa: N806
"""Muse Glimmer oQ A8 (INT8-activation) prefill module for oMLX.

This module implements the Muse Glimmer-specific surfaces for the oQ A8
prefill optimization, reusing the model-agnostic helpers from
qwen35_oq_a8.py. It patches the MLP class for shared activation
quantization (SwiGLU gate+up) and installs linear wrappers for attention
projections.

The oQ A8 kernels are identical to Qwen's -- NAX int8xint8->int32
with a FP32 affine correction. What differs per model is only the module
surfaces: Muse Glimmer has a SwiGLU MLP (same as Qwen) and standard 
MHA attention with an extra gate_proj on the attention output.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from .qwen35_oq_a8 import (
    _CONFIG_ATTR,
    _MIN_TOKENS_DEFAULT,
    OqA8Config,
    _kernels_available,
    _prefill_linear_backend,
    _swiglu,
    _tag_modules,
    oq_a8_mlp,
)

logger = logging.getLogger(__name__)

_MLP_PATCHED_TARGETS = set()
_LINEAR_PATCHED = False

# Muse Glimmer's MLP is in the vendored mlx_vlm module; it uses SwiGLU
# (gate_proj, up_proj, down_proj), matching the Qwen MLP surface that
# qwen35_oq_a8.oq_a8_mlp expects.
#
# Two MLP classes exist for Muse Glimmer depending on the engine path:
# - mlx_vlm.models.muse_glimmer.language.MLP — used by VLMBatchedEngine
#   (interactive/vlm path).
# - mlx_lm.models.muse_glimmer.MLP — used by BatchedEngine when
#   force_lm=True (benchmark path, since Muse Glimmer is not in
#   VLM_NATIVE_TEXT_MODEL_TYPES).
# Both must be patched so the int8 kernel fires regardless of engine.
_MUSE_GLIMMER_MLP_TARGETS = (
    ("mlx_vlm.models.muse_glimmer.language", "MLP"),
    ("mlx_lm.models.muse_glimmer", "MLP"),
)


def muse_glimmer_oq_a8_mlp(mlp: Any, x: mx.array) -> mx.array | None:
    """SwiGLU MLP with Stage A shared between gate and up.

    Uses the model-agnostic oq_a8_mlp logic with the Muse Glimmer SwiGLU
    activation (identical to Qwen's). Returns None when any of the three
    projections is not eligible.
    """
    return oq_a8_mlp(mlp, x, _swiglu())


def _make_patched_muse_mlp(orig_call):
    def patched(self, x: mx.array, *args, **kwargs):
        config = getattr(self, _CONFIG_ATTR, None)
        if (
            config is None
            or x.ndim < 3
            or x.shape[-2] <= 1
            or x.shape[-2] < config.min_tokens
        ):
            return orig_call(self, x, *args, **kwargs)
        # MTP verify forwards are decode-shaped (1 token) and must never route.
        # Use kwargs only for target_verify to avoid fragile positional mapping.
        target_verify = bool(kwargs.get("target_verify", False))
        if target_verify:
            return orig_call(self, x, *args, **kwargs)
        out = muse_glimmer_oq_a8_mlp(self, x)
        if out is None:
            return orig_call(self, x, *args, **kwargs)
        return out

    return patched


def _patch_muse_mlp_class(module_name: str, class_name: str) -> bool:
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False
    cls = getattr(module, class_name, None)
    if cls is None:
        return False
    if getattr(cls, "_omlx_muse_glimmer_oq_a8_patched", False):
        return True
    orig = cls.__call__
    cls.__call__ = _make_patched_muse_mlp(orig)
    cls._omlx_muse_glimmer_oq_a8_patched = True
    cls._omlx_muse_glimmer_oq_a8_original_call = orig
    return True


def apply_muse_glimmer_oq_a8_patch(
    model: Any = None,
    min_tokens: int | None = None,
) -> bool:
    """Install shared wrappers and tag only this model with its configuration.

    1. Installs MLP class wrappers for shared Stage A.
    2. Installs linear wrappers for attention projections via
       muse_glimmer_oq_a8_linear.
    3. Registers the model-agnostic _prefill_linear_backend (returns None on
       decline, letting the wrapper fall back) as the muse_glimmer prefill
       backend.
    """
    global _MLP_PATCHED_TARGETS, _LINEAR_PATCHED

    if not _kernels_available():
        logger.debug("muse_glimmer oq_a8: kernels unavailable; patch skipped")
        return False

    floor = _MIN_TOKENS_DEFAULT
    if min_tokens is not None and int(min_tokens) >= 1:
        floor = int(min_tokens)
    config = OqA8Config(min_tokens=floor)

    tagged = _tag_modules(model, config) if model is not None else 0

    newly_patched_mlp = 0
    for module_name, class_name in _MUSE_GLIMMER_MLP_TARGETS:
        target = (module_name, class_name)
        if target not in _MLP_PATCHED_TARGETS:
            if _patch_muse_mlp_class(module_name, class_name):
                _MLP_PATCHED_TARGETS.add(target)
                newly_patched_mlp += 1

    if not _LINEAR_PATCHED:
        try:
            from omlx.patches.muse_glimmer_oq_a8_linear import (
                register_muse_glimmer_prefill_linear_backend,
            )

            # Register the model-agnostic first-refusal backend.  It returns
            # None on decline (not linear(x) like oq_a8_linear), so the wrapper
            # falls back to super().__call__ without re-entering itself.
            register_muse_glimmer_prefill_linear_backend(_prefill_linear_backend)
            _LINEAR_PATCHED = True
        except Exception:
            logger.debug(
                "muse_glimmer oq_a8: linear backend registration failed",
                exc_info=True,
            )

    # The reclassification is per-instance, so it must run on every model
    # load, not just the first one; already-wrapped linears fail the
    # exact-type check, which keeps repeat calls idempotent.
    installed = 0
    if _LINEAR_PATCHED and model is not None:
        try:
            from omlx.patches.muse_glimmer_oq_a8_linear import (
                apply_muse_glimmer_oq_a8_prefill_linear_patch,
            )

            installed = apply_muse_glimmer_oq_a8_prefill_linear_patch(model)
        except Exception:
            logger.debug(
                "muse_glimmer oq_a8: per-model linear reclassification failed",
                exc_info=True,
            )

    if _MLP_PATCHED_TARGETS or _LINEAR_PATCHED:
        logger.info(
            "muse_glimmer oQ A8 prefill: mlp_patched_count=%s, backend_registered=%s, "
            "linear_wrapped_count=%d, modules_tagged=%d, min_tokens=%d",
            len(_MLP_PATCHED_TARGETS),
            bool(_LINEAR_PATCHED),
            installed,
            tagged,
            config.min_tokens,
        )
        if newly_patched_mlp > 0:
            logger.info("muse_glimmer oQ A8 prefill: newly patched %d MLP targets", newly_patched_mlp)

    return bool(_MLP_PATCHED_TARGETS) or _LINEAR_PATCHED
