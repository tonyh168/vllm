# SPDX-License-Identifier: Apache-2.0

"""LogitsProcessor layer - Phase 1: passthrough."""

from vllm.model_executor.layers.logits_processor import LogitsProcessor


def register_logits_layers() -> None:
    """Register LogitsProcessor OOT layer (no-op inheritance for Phase 1)."""

    @LogitsProcessor.register_oot
    class RefLogitsProcessor(LogitsProcessor):
        pass
