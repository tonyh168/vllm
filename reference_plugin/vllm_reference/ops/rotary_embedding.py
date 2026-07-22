# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding ops - Phase 1: delegate to forward_cuda."""

import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.dual_chunk_rope import (
    DualChunkRotaryEmbedding,
)


def register_rotary_ops() -> None:
    """Register all rotary embedding OOT ops."""

    @RotaryEmbeddingBase.register_oot
    class RefRotaryEmbeddingBase(RotaryEmbeddingBase):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)

    @ApplyRotaryEmb.register_oot
    class RefApplyRotaryEmb(ApplyRotaryEmb):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)

    @DualChunkRotaryEmbedding.register_oot
    class RefDualChunkRotaryEmbedding(DualChunkRotaryEmbedding):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)
