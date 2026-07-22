# SPDX-License-Identifier: Apache-2.0

"""Mamba layers - Phase 1: passthrough."""

from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention


def register_mamba_layers() -> None:
    """Register Mamba OOT layers (no-op inheritance for Phase 1)."""

    @MambaMixer.register_oot
    class RefMambaMixer(MambaMixer):
        pass

    @MambaMixer2.register_oot
    class RefMambaMixer2(MambaMixer2):
        pass

    @GatedDeltaNetAttention.register_oot
    class RefGatedDeltaNetAttention(GatedDeltaNetAttention):
        pass
