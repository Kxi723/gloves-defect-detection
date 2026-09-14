# Glove Defect Detection System

Three glove defects found with classical image processing (OpenCV and NumPy, no
training data). Damage by fold, dirty, and tearing at the fingertip.

## Run it

```bash
C:\Tool\python\python.exe studio.py
```

The launcher has one tile per defect, each showing a real glove from `gloves/`
cut out with the detector's own segmentation (cached in `output/.landing/`).
Picking one runs every photo in `gloves/` straight away and replays the detector one step at a time, so the preprocessing,
the segmentation and the analysis are on screen rather than implied.

```
capture -> resize -> white balance -> bilateral
        -> background distance -> cue vote -> cleanup -> largest component
        -> backdrop surface -> GrabCut -> glove found
        -> the analysis steps of that defect -> verdict
```

Fold plays 22 steps, dirty 18 and tear 19. The source photo stays on the left
and the current step sweeps in on the right. Every photo is processed once. The
folder list on the right keeps each finished photo with its verdict, and clicking
one opens it for reading on its verdict frame without running the detector again.
The live run holds meanwhile, and clicking the live row (or play, or Space)
resumes it where it was left. The pipeline strip underneath scrubs to any step,
the arrow buttons and keys walk through the finished photos, and SAVE writes the
frame on screen to `output/<defect>/`. Esc returns to the launcher.

Frames play at a fixed pace. The detector itself runs at full speed in a
background thread, about 0.6s a photo on its own and 1.3s with every frame
rendered, so it always stays ahead of the playback.

Photos live in `gloves/`, named `Fold_*`, `Dirty_*` or `TearFinger_*` after the
defect they should show. Use the full interpreter path, a bare `python` hits the
Microsoft Store stub.

## Layout

```
studio.py         entry point
ui/qc.py          the QC floor look and the line screen      ui/studio.py   the inspect screen
ui/neon.py        screen scaling and colour helpers
pipeline.py       step tracing, calls each detector's own functions
detectors/        one self contained file per defect
runner.py         headless harness, used by evaluate.py
gloves/           the photos
```

**Each file in `detectors/` is standalone.** It carries its own preprocessing,
segmentation, measurement helpers and thresholds, and imports nothing from the
project. `pipeline.py` only calls into those modules and captures what they
produce, so the verdict on screen always comes from the detector's own `detect`.

## How the glove is separated from the backdrop

`segment_glove` in each module runs three passes.

1. **Cue vote.** Six ways of calling background (a border sampled background
   model, Lab distance from the backdrop, three Otsu splits on HSV, and a texture
   energy map) are each scored on area, compactness, how much filling the outline
   needed, and how well that outline lands on real image edges. The edge term is
   what stops a mask whose boundary wanders through flat cloth from winning.
2. **Backdrop surface.** With the glove roughly located, a low order surface is
   fitted per Lab channel to the pixels that are actually backdrop. Sampling only
   the frame border treats the cloth as one flat colour, but the lamp puts a
   bright halo around the glove and lets the corners fall away, so the middle of
   the cloth sits far from the border median and reads as glove. Fitting the
   falloff predicts it instead, and only real material is left over.
3. **GrabCut.** The boundary is re cut at reduced resolution from the colour
   statistics of both sides, which is what drops a sleeve or a forearm that the
   thresholds had joined onto the glove.

## How each defect is decided

**Fold.** Three channels look for a dark line in the palm (shading ridge, weave
bend, chroma residual). A shadow gate drops anything brighter than the glove,
which is glare on a flat sheet. A span gate then asks for the line to run across
the palm, at least 1.33 palm radii once extended along its own direction, because
a fold is pressed across the palm while a natural wrinkle or a knit stripe is a
short line. On these photos the shortest real fold spans 1.44R and the longest
wrinkle 1.23R, and the gate sits between them.

**Dirty.** Blobs whose lightness is off from the glove itself, deep enough inside
the glove to rule out rim shading, are kept when they are texture free (something
covers the weave) or carry a hue the glove never shows.

**Tearing at fingertip.** Fingertips come from the hand skeleton. Holes, show
through patches and contour notches only count when they land inside a ring
around a tip.

## Current state

| detector | agrees with the filename |
|---|---|
| Damage by Fold | 15 of 15 |
| Dirty | 15 of 15 |
| Tearing at Fingertip | 15 of 15 |

Every detector is run on every photo, so a clean photo firing counts as a miss.
The 15 photos are the whole data set, used both to tune and to demo, so this is
not a held out number. Thresholds sit on measured gaps rather than on a single
image wherever that was possible. There are no undamaged glove photos yet, so a
false alarm can only be caught against the other two defect types.

**Photograph the glove flat and empty, whole glove inside the frame.**

## Adding a detector

Write one self contained file in `detectors/` exposing `Config`, `preprocess`,
`segment_glove` and `detect`, add it to `DEFECTS` in `detectors/__init__.py`, and
add a card for it to `DEFECTS` in `pipeline.py` together with its analysis steps.

Express thresholds as fractions of the palm radius or of the glove area, never in
pixels, so they survive a change of camera or framing. Take reference levels from
the glove in the photo itself (robust median and MAD) rather than hard coding
them, so they adapt to colour, material and lighting.
