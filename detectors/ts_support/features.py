

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]


@dataclass
class DefectResult:


    defect_found: bool
    defect_type: str
    locations: List[BBox] = field(default_factory=list)
    score: float = 0.0
    details: str = ""
    debug_mask: Optional[np.ndarray] = None
    analysis_mask: Optional[np.ndarray] = None
