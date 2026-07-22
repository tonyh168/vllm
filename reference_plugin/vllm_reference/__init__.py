# SPDX-License-Identifier: Apache-2.0

"""vLLM Reference Plugin - Hardware-agnostic reference implementation.

Phase 1: Delegates all operations to existing CUDA implementations.
Phase 2: Will replace CUDA-specific ops with PyTorch/Triton implementations.
"""


def register() -> str | None:
    """Platform plugin entry point.

    Returns the fully qualified class name of the ReferencePlatform,
    or None if CUDA is not available (since Phase 1 still runs on CUDA).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    return "vllm_reference.platform.ReferencePlatform"


def register_ops() -> None:
    """General plugin entry point. Registers all OOT CustomOps and
    PluggableLayer replacements."""
    from vllm.model_executor.custom_op import CustomOp

    # Phase 1: Override forward_oot at the base class level to delegate
    # to forward_cuda for ALL CustomOps. This is the simplest way to
    # make every op work on our OOT platform without individually
    # registering each subclass.
    def forward_oot_via_cuda(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    CustomOp.forward_oot = forward_oot_via_cuda
