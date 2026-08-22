from __future__ import annotations
import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import List

@dataclass(frozen=True)
class DefectSpec:
    key: str
    label: str
    implemented: bool = True
    owner: str = ""

    def load(self) -> ModuleType:
        return importlib.import_module(f"{__name__}.{self.key}")

# The order shown in the menu.
DEFECTS: List[DefectSpec] = [
    DefectSpec("jason_damage_by_fold", "Damage by Fold - Jason Lai Kwang Xi"),
    DefectSpec("jason_dirty", "Dirty - Jason Lai"),
    DefectSpec("sheng_finger_not_enough", "Finger Not Enough - Sheng"),
    DefectSpec("sheng_improper_roll", "Improper Roll - Sheng"),
    DefectSpec("oversized", "Oversize - Yung Cee"),
    DefectSpec("sheng_plastic_contamination", "Plastic Contamination - Sheng"),
    DefectSpec("stain", "Stain - Yung Cee"),
    DefectSpec("jason_tearing_at_finger", "Tearing(fingertip) - Jason Lai Kwang Xi"),
    DefectSpec("wrinkle", "Wrinkle - Yung Cee"),
]
