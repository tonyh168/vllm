# SPDX-License-Identifier: Apache-2.0

"""MLA layers - Phase 1: passthrough."""

from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.deepseek_v4_attention import (
    DeepseekV4MultiHeadLatentAttentionWrapper,
)


def register_mla_layers() -> None:
    """Register MLA OOT layers (no-op inheritance for Phase 1)."""

    @MultiHeadLatentAttentionWrapper.register_oot
    class RefMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
        pass

    @DeepseekV4MultiHeadLatentAttentionWrapper.register_oot
    class RefDeepseekV4MLA(DeepseekV4MultiHeadLatentAttentionWrapper):
        pass
