from __future__ import annotations

from .base import BaseAdapter


class MIDUSAdapter(BaseAdapter):
    """Placeholder adapter for MIDUS public portal files."""

    def load_raw(self):
        raise NotImplementedError("Dataset-specific adapter not yet implemented.")

    def to_long_person_trait(self):
        raise NotImplementedError("Dataset-specific adapter not yet implemented.")
