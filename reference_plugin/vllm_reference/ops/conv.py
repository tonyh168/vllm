# SPDX-License-Identifier: Apache-2.0

"""Conv ops - Phase 1: delegate to forward_cuda."""

import torch

from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer


def register_conv_ops() -> None:
    """Register conv OOT ops."""

    @Conv2dLayer.register_oot
    class RefConv2dLayer(Conv2dLayer):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)

    @Conv3dLayer.register_oot
    class RefConv3dLayer(Conv3dLayer):
        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward_cuda(x)
