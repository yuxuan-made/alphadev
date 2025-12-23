"""Universe loader.

Universe files are stored using the same layout as other Features, so this loader
is a thin wrapper around FeatureLoader.

Expected output column: 'in_universe' (bool/0/1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ...alpha.features import Feature
from .feature_loader import FeatureLoader


class UniverseLoader(FeatureLoader):
    def __init__(self, universe: Feature, feature_dir: Optional[Path] = None):
        super().__init__(feature=universe, feature_dir=feature_dir)

    def get_name(self) -> str:
        return f"UniverseLoader({self.feature.get_name()})"


__all__ = ["UniverseLoader"]
