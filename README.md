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

`app.py` is the UI. It has two pages. The inspect page picks a defect on
the left, the photos in the middle and lists a result per photo on the
right. The compare page shows the original and the detected photo side by
side at the same scale, with a filmstrip of the run underneath, and it can
save the pair as one PNG for the report. Open it by double-clicking a
result row or by clicking the small preview. Left and Right walk through
the run and Esc goes back.

`test_pipeline.py` runs a whole folder and writes
report figures (original | mask | detection) to `output/`. `evaluate.py`
prints precision and recall per detector, reading the ground truth from
the filename (`kxi_latex_3.jpeg` → latex → should find `damage_by_fold`).

Photos live in `gloves/`. Use the full interpreter path — a bare `python`
hits the Microsoft Store stub.

## Layout

```
app.py          UI, two pages        ui/     UI toolkit + compare page
detectors/      one self-contained file per defect
runner.py       local test host      test_pipeline.py, evaluate.py
```

**Each file in `detectors/` is standalone.** It carries its own
preprocessing, segmentation, measurement helpers and thresholds, and
imports nothing from this project — so it can be dropped into the group's
shared UI, or any other host, on its own.

That means the same front-end code appears in every detector file. This is
deliberate. The group photographs gloves under different angles and
lighting, so each defect needs to tune its own preprocessing and
segmentation; one shared version would force a single compromise on all of
them. The cost is that a fix to shared plumbing has to be applied per file.

`runner.py` is NOT part of that contract. It is the local harness that
`app.py`, `test_pipeline.py` and `evaluate.py` share, so the three behave
identically: it drives detector modules, collects results, and draws the
annotated photos and report figures.

## Adding a detector

Write one self-contained file in `detectors/` exposing one function, then
append a `DefectSpec` to `DEFECTS` in `detectors/__init__.py`. Nothing else
in the project changes.

```python
def detect(image) -> DefectResult: ...
```

`image` is a raw BGR photo straight from `cv2.imread`; the module does its
own preprocessing and segmentation. Expose `Config`, `preprocess` and
`segment_glove` as well and `runner.py` will drive that front end itself,
which saves running it twice and lets a report figure show the mask your
detector actually saw.

Two conventions that matter, because the marker will use photos we have
never seen:

- Express thresholds as fractions of the palm radius or glove area, never
  in pixels, so they survive a change of camera or framing.
- Take reference levels from the glove in the photo itself (robust median
  and MAD) rather than hard-coding them, so they adapt to colour, material
  and lighting.

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
