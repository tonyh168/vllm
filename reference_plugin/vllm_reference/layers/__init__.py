# SPDX-License-Identifier: Apache-2.0

"""Reference plugin layers.

Phase 1: No PluggableLayer replacements needed since the in-tree
classes already work correctly on CUDA.

Phase 2: Individual layer files will contain hardware-agnostic replacements.
"""


def register_all_layers() -> None:
    """Phase 2 placeholder: register individual layer replacements."""
    pass
