# SPDX-License-Identifier: Apache-2.0

"""Mamba ops - Phase 1: delegate to forward_cuda."""

from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.layers.mamba.gdn_linear_attn import ChunkGatedDeltaRule


def register_mamba_ops() -> None:
    """Register Mamba-related OOT ops."""

    @Mixer2RMSNormGated.register_oot
    class RefMixer2RMSNormGated(Mixer2RMSNormGated):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)

    @ShortConv.register_oot
    class RefShortConv(ShortConv):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)

    @ChunkGatedDeltaRule.register_oot
    class RefChunkGatedDeltaRule(ChunkGatedDeltaRule):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)
