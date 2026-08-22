"""Support types and image-processing helpers used by TS detectors.

There is intentionally no shared execution pipeline in this package. Each
detector imports only the helper functions it needs and owns its complete
processing sequence.
"""

from .config import PipelineConfig, get_config
from .features import BBox, DefectResult

__all__ = ["BBox", "DefectResult", "PipelineConfig", "get_config"]
