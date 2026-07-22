# SPDX-License-Identifier: Apache-2.0

"""Attention ops - Phase 1: delegate to forward_cuda."""

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.attention.static_sink_attention import (
    StaticSinkAttention,
)


def register_attention_ops() -> None:
    """Register attention-related OOT ops."""

    @MMEncoderAttention.register_oot
    class RefMMEncoderAttention(MMEncoderAttention):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)

    @StaticSinkAttention.register_oot
    class RefStaticSinkAttention(StaticSinkAttention):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)
