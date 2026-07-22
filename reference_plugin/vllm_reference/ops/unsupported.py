# SPDX-License-Identifier: Apache-2.0

"""Unsupported ops - registered to raise NotImplementedError."""

from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8


def register_unsupported_ops() -> None:
    """Register ops that are explicitly unsupported in the reference plugin."""

    @SparseAttnIndexer.register_oot
    class RefSparseAttnIndexer(SparseAttnIndexer):
        def forward_oot(self, *args, **kwargs):
            raise NotImplementedError(
                "[vllm-reference] SparseAttnIndexer is not supported. "
                "It requires C++ CUDA kernels (top_k_per_row_*, persistent_topk) "
                "that have no PyTorch/Triton alternative."
            )

    @QuantFP8.register_oot
    class RefQuantFP8(QuantFP8):
        def forward_oot(self, *args, **kwargs):
            raise NotImplementedError(
                "[vllm-reference] QuantFP8 is not supported. "
                "The reference plugin only supports non-quantized inference."
            )
