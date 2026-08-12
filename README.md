# Gloves Defect Detection (GDD)

CT036-3-IPPR group assignment. Classical image processing only (OpenCV +
NumPy) — no deep learning, Haar cascades, or template matching.

> Work in progress. Three of the twelve defects are implemented; the
> README will be rewritten once the full set is in.

## Run it

```bash
C:\Tool\python\python.exe app.py
```

```bash
C:\Tool\python\python.exe test_pipeline.py
```

```bash
C:\Tool\python\python.exe evaluate.py
```

`app.py` is the UI. `test_pipeline.py` runs a whole folder and writes
report figures (original | mask | detection) to `output/`. `evaluate.py`
prints precision and recall per detector, reading the ground truth from
the filename (`kxi_latex_3.jpeg` → latex → should find `damage_by_fold`).

Photos live in `gloves/`. Use the full interpreter path — a bare `python`
hits the Microsoft Store stub.

## Layout

```
app.py          UI                  ui/     UI toolkit
detectors/      one file per defect  gdd/   shared engine
test_pipeline.py, evaluate.py        entry points
```

**`detectors/`** answers *does this glove have this defect*. **`gdd/`**
answers *where is the glove, how big is it, how dark is this patch* — the
measurements and the pipeline that every detector shares.

| `gdd/` | |
|---|---|
| `config.py` | every tunable threshold |
| `preprocessing.py` | resize, white balance, denoise |
| `segmentation.py` | five-cue glove/background separation |
| `features.py` | measurement helpers detectors build on |
| `pipeline.py` | `GloveInspector`: runs the chain, draws results |

## Adding a detector

Write one file in `detectors/` with one function, then append a
`DefectSpec` to `DEFECTS` in `detectors/__init__.py`. Nothing else changes.

```python
def detect(image, segmentation, config) -> DefectResult: ...
```

`image` is the preprocessed BGR photo, `segmentation` carries the glove
mask, contour and area, `config` is the shared `PipelineConfig`. Build on
`gdd/features.py` instead of reimplementing geometry — it has the palm
centre and radius, glove interior, fingertip locations, convexity defects,
hole finding, robust statistics and local texture energy.

Two conventions that matter, because the marker will use photos we have
never seen:

- Express thresholds as fractions of the palm radius or glove area, never
  in pixels, so they survive a change of camera or framing.
- Take reference levels from the glove in the photo itself (see
  `robust_stats`) rather than hard-coding them, so they adapt to colour,
  material and lighting.

## Current state

| detector | file | precision | recall |
|---|---|---|---|
| Damage by Fold | `detectors/damage_by_fold.py` | 67% | 80% |
| Dirty | `detectors/dirty.py` | 83% | 100% |
| Tearing (fingertip) | `detectors/tearing_at_finger.py` | 100% | 20% |

Measured by `evaluate.py` over 15 photos. Two caveats it prints itself:
there are no undamaged-glove photos yet, so precision only reflects
confusion with the other defect types; and the low tearing recall is a
capture problem, not an algorithm one — four of the five torn gloves were
photographed while worn, where a fingertip tear barely changes the
silhouette.

**Photograph the glove flat and empty, whole glove inside the frame.**

## Working on it

Run `evaluate.py` after every change to see whether accuracy moved, then
`test_pipeline.py` to check the boxes landed in the right places. Numbers
alone mislead — a mask can score a plausible area while covering the wrong
region — and pictures alone cannot tell you whether a change helped
overall.
