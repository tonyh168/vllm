# SPDX-License-Identifier: Apache-2.0

"""FusedMoE ops - Phase 1: delegate to forward_cuda."""

import torch

from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopk,
)


def register_moe_ops() -> None:
    """Register MoE OOT ops."""

    @UnquantizedFusedMoEMethod.register_oot
    class RefUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)

    @GroupedTopk.register_oot
    class RefGroupedTopk(GroupedTopk):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)
