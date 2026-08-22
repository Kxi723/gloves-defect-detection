"""Output types shared only by TS's three detector-owned workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class DefectResult:
    """Result returned by each TS ``detect(image)`` function.

    The group dispatcher reads this uniform structure without taking control
    of the detector's preprocessing, segmentation or decision rules.
    """

    defect_found: bool
    defect_type: str
    locations: List[BBox] = field(default_factory=list)
    score: float = 0.0  # 0..1 rule-based evidence score, not accuracy
    details: str = ""  # measured evidence displayed by the interface
    debug_mask: Optional[np.ndarray] = None  # accepted defect candidates
    analysis_mask: Optional[np.ndarray] = None  # detector-owned glove mask
