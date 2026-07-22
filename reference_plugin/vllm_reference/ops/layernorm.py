# SPDX-License-Identifier: Apache-2.0

"""LayerNorm ops - Phase 1: delegate to forward_cuda."""

import torch

from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm,
    RMSNorm,
    RMSNormGated,
)


def register_layernorm_ops() -> None:
    """Register all layernorm OOT ops."""

    @RMSNorm.register_oot
    class RefRMSNorm(RMSNorm):
        def forward_oot(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            return self.forward_cuda(x, residual)

    @GemmaRMSNorm.register_oot
    class RefGemmaRMSNorm(GemmaRMSNorm):
        def forward_oot(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            return self.forward_cuda(x, residual)

    @RMSNormGated.register_oot
    class RefRMSNormGated(RMSNormGated):
        def forward_oot(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x, gate)
