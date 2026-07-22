# SPDX-License-Identifier: Apache-2.0

"""Linear layers - Phase 1: passthrough."""

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


def register_linear_layers() -> None:
    """Register linear OOT layers (no-op inheritance for Phase 1)."""

    @ReplicatedLinear.register_oot
    class RefReplicatedLinear(ReplicatedLinear):
        pass

    @ColumnParallelLinear.register_oot
    class RefColumnParallelLinear(ColumnParallelLinear):
        pass

    @RowParallelLinear.register_oot
    class RefRowParallelLinear(RowParallelLinear):
        pass
