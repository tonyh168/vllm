# SPDX-License-Identifier: Apache-2.0

"""Reference plugin ops.

Phase 1: forward_oot -> forward_cuda delegation is handled globally
via the monkey-patch in __init__.py.

Phase 2: Individual op files will contain PyTorch/Triton replacements.
"""


def register_all_ops() -> None:
    """Phase 2 placeholder: register individual op replacements."""
    pass
