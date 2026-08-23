# Gloves Defect Detection (GDD)

CT036-3-IPPR group assignment implemented with classical image processing
(OpenCV + NumPy). The project does not use TensorFlow, Haar cascades or
template/pattern matching.

## Setup and run

```bash
python -m pip install -r requirements.txt
python app.py
```

The main UI lets the user choose a defect, choose one or more glove photos,
run that detector and inspect the confidence/evidence for each image.
`Compare with original` shows the original and annotated result side by side.

## Project structure

```text
app.py                    GUI
runner.py                 common detector host and annotation
ui/                       interface components
detectors/                one module per defect
detectors/*_support/      member-specific preprocessing/features/config
gloves/                   group test images
img/                      additional organised image sets
```

A detector exposes `detect`. A detector that also exposes `Config`,
`preprocess` and `segment_glove` is driven through its own preprocessing and
segmentation settings by `runner.py`. This lets each member tune their own
image-processing thresholds without changing another member's detector.

## Tan Yik Ting detectors

- `detectors/tearing.py`
- `detectors/incomplete_beading.py`
- `detectors/spotting.py`
- support code: `detectors/TanYikTing_support/`

These detectors use colour-space conversion, thresholding, morphology,
connected components, contour/shape measurements and scale-normalised
features. Their configuration is local to `TanYikTing_support/config.py`.

## Test images

Photos placed directly in `gloves/` are loaded automatically by the current
GUI. The submitted project should keep the test images together with the
source code so the detection results can be reproduced.
