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

# The three detectors this project runs.
DEFECTS: List[DefectSpec] = [
    DefectSpec("jason_damage_by_fold", "Damage by Fold"),
    DefectSpec("jason_dirty", "Dirty"),
    DefectSpec("jason_tearing_at_finger", "Tearing at Fingertip"),
]
