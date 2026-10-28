"""Offline plate dataset toolkit. Arrays use BGR and quads use BR, BL, TL, TR."""
__version__ = "1.0.0"

from .rules import PROVINCES, LETTERS, SMALL_THIRD, valid_text

__all__ = ["PROVINCES", "LETTERS", "SMALL_THIRD", "valid_text"]
