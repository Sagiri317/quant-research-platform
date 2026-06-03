"""A-share label builders."""

from . import build_labels
from .forward_returns import add_forward_return_labels

__all__ = ["add_forward_return_labels", "build_labels"]
