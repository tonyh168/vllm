# SPDX-License-Identifier: Apache-2.0

"""Reference Platform - Phase 1: delegates to CUDA implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms.cuda import CudaPlatformBase, NvmlCudaPlatform
from vllm.platforms.interface import PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ReferencePlatform(NvmlCudaPlatform):
    """Reference platform plugin.

    Phase 1: Inherits from NvmlCudaPlatform to reuse all CUDA logic.
    The only difference is _enum = OOT so CustomOp dispatches to
    forward_oot (which we monkey-patch to forward_cuda).

    Phase 2: Will override methods to use PyTorch/Triton implementations.
    """

    _enum = PlatformEnum.OOT
    device_name: str = "reference"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"

    @classmethod
    def is_cuda_alike(cls) -> bool:
        return True

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # Phase 1: reuse CudaCommunicator directly
        return (
            "vllm.distributed.device_communicators"
            ".cuda_communicator.CudaCommunicator"
        )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads=None,
    ) -> str:
        # Phase 1: delegate to parent CUDA implementation
        return super().get_attn_backend_cls(
            selected_backend, attn_selector_config, num_heads
        )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # Phase 1: delegate to parent
        super().check_and_update_config(vllm_config)

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        raise NotImplementedError(
            f"[vllm-reference] Quantization method '{quant}' is not supported. "
            "The reference plugin only supports non-quantized inference."
        )
