"""Deprecated compatibility shim.

Use `core.qwen_audio_2_utils` for Qwen2-Audio utilities.
"""

from warnings import warn

warn(
    "core.qwen_utils is deprecated; import core.qwen_audio_2_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

from core.qwen_audio_2_utils import *  # noqa: F401,F403
