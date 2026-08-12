"""One module per defect the system offers, plus the menu registry.

`app.py` reads :data:`DEFECTS`, draws one menu entry per spec, and imports
the matching module only when the user clicks it. A module needs nothing
more than

    detect(image, segmentation, config) -> DefectResult

which is exactly the plugin contract `GloveInspector.register_detector`
uses, so the same file works in the GUI, in the batch runner and in the
evaluator.

Adding a defect means writing one file here and appending one line to
:data:`DEFECTS` (plus a config block in `gdd/config.py` if it needs
tunables). Nothing else in the project has to change, which is the whole
point of the split: detector files never import each other, so several
people can work without touching shared code. The measurements they all
need live in `gdd/features.py`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import List


@dataclass(frozen=True)
class DefectSpec:
    """One entry of the defect menu."""
    key: str                    # registry name, also the module file name
    label: str                  # text shown in the menu
    implemented: bool = True    # False -> menu marks it as pending
    owner: str = ""             # who is writing it (for the pending ones)

    def load(self) -> ModuleType:
        """Import this defect's module (done lazily, on click)."""
        return importlib.import_module(f"{__name__}.{self.key}")


# Order here is the order shown in the menu.
#
# This project covers three defects. The full system has twelve, the other
# nine belonging to other members; each is added by dropping one file into
# this package and appending one DefectSpec below, with no change anywhere
# else. Entries can be listed before their file exists by passing
# implemented=False, which greys the menu entry out.
DEFECTS: List[DefectSpec] = [
    DefectSpec("damage_by_fold", "Damage by Fold"),
    DefectSpec("dirty", "Dirty"),
    DefectSpec("tearing_at_finger", "Tearing(fingertip)"),
]

