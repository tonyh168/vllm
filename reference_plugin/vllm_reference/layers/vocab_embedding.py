# SPDX-License-Identifier: Apache-2.0

"""Vocab embedding layers - Phase 1: passthrough."""

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


def register_embedding_layers() -> None:
    """Register embedding OOT layers (no-op inheritance for Phase 1)."""

    @VocabParallelEmbedding.register_oot
    class RefVocabParallelEmbedding(VocabParallelEmbedding):
        pass

    @ParallelLMHead.register_oot
    class RefParallelLMHead(ParallelLMHead):
        pass
