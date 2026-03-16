from __future__ import annotations

from .base import BaseAdapter


class AddHealthAdapter(BaseAdapter):
    """Placeholder adapter for Add Health public-use subset."""

    def load_raw(self):
        raise NotImplementedError("Dataset-specific adapter not yet implemented.")

    def to_long_person_trait(self):
        raise NotImplementedError("Dataset-specific adapter not yet implemented.")
