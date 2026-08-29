"""Optional integrations for external embodied inference runtimes."""

from .embodied_cpp import EmbodiedCppVlaBackend
from .http_json import HttpJsonBackend
from .smolvla import SmolVLABackend

__all__ = ["EmbodiedCppVlaBackend", "HttpJsonBackend", "SmolVLABackend"]
