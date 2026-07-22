# SPDX-License-Identifier: Apache-2.0

"""Activation ops - Phase 1: delegate to forward_cuda."""

import torch

from vllm.model_executor.layers.activation import (
    GELU,
    XIELU,
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    GeluAndMulSparse,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    ReLUSquaredActivation,
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluOAIAndMul,
    SwigluStepAndMul,
)


def register_activation_ops() -> None:
    """Register all activation OOT ops."""

    @SiluAndMul.register_oot
    class RefSiluAndMul(SiluAndMul):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @SiluAndMulWithClamp.register_oot
    class RefSiluAndMulWithClamp(SiluAndMulWithClamp):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @MulAndSilu.register_oot
    class RefMulAndSilu(MulAndSilu):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @FatreluAndMul.register_oot
    class RefFatreluAndMul(FatreluAndMul):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @GeluAndMul.register_oot
    class RefGeluAndMul(GeluAndMul):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @GeluAndMulSparse.register_oot
    class RefGeluAndMulSparse(GeluAndMulSparse):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @GELU.register_oot
    class RefGELU(GELU):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @NewGELU.register_oot
    class RefNewGELU(NewGELU):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @FastGELU.register_oot
    class RefFastGELU(FastGELU):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @QuickGELU.register_oot
    class RefQuickGELU(QuickGELU):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @ReLUSquaredActivation.register_oot
    class RefReLUSquaredActivation(ReLUSquaredActivation):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @SwigluOAIAndMul.register_oot
    class RefSwigluOAIAndMul(SwigluOAIAndMul):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @SwigluStepAndMul.register_oot
    class RefSwigluStepAndMul(SwigluStepAndMul):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @XIELU.register_oot
    class RefXIELU(XIELU):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)
