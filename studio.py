"""Entry point for the Glove Defect Detection System.

    C:\\Tool\\python\\python.exe studio.py

Three defects, every photo in gloves/, and the pipeline played back one step
at a time. The original twelve defect assignment UI still lives in app.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.studio import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
