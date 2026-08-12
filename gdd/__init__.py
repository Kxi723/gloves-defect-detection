"""
GDD - Gloves Defect Detection (CT036-3-IPPR group assignment).

`gdd` is the shared engine: preprocessing, segmentation, the measurement
helpers every detector builds on, and the inspection pipeline. The
detectors themselves live one-per-file in the top-level ``detectors``
package, so twelve defects written by four people never touch each other's
code.

Only classical image processing is used (OpenCV + NumPy). No deep
learning, no Haar cascades, no template matching.
"""

from gdd.config import PipelineConfig, get_config
from gdd.preprocessing import preprocess, normalize_illumination
from gdd.segmentation import SegmentationResult, segment_glove
from gdd.features import DefectResult, BBox
from gdd.pipeline import GloveInspector, InspectionReport, assess_capture

__all__ = [
    "PipelineConfig",
    "get_config",
    "preprocess",
    "normalize_illumination",
    "SegmentationResult",
    "segment_glove",
    "DefectResult",
    "BBox",
    "GloveInspector",
    "InspectionReport",
    "assess_capture",
]
