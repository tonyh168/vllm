# SPDX-License-Identifier: Apache-2.0

"""Reference Communicator - Phase 1: inherits CudaCommunicator directly.

Phase 2 will replace this with pure torch.distributed implementation.
"""

from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)


class ReferenceCommunicator(CudaCommunicator):
    """Phase 1: Direct passthrough to CudaCommunicator.

    In Phase 2, this will inherit from DeviceCommunicatorBase and use
    only torch.distributed primitives.
    """

    pass
