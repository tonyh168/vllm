# SPDX-License-Identifier: Apache-2.0

"""FusedMoE layer - Phase 1: passthrough."""

from vllm.model_executor.layers.fused_moe.layer import FusedMoE


def register_fused_moe_layers() -> None:
    """Register FusedMoE OOT layer (no-op inheritance for Phase 1)."""

    @FusedMoE.register_oot
    class RefFusedMoE(FusedMoE):
        pass
