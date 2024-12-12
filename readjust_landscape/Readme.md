# Readjust KTP Orientation

This repository contains code for readjust the orientation of KTP. This code check wheter KTP is on the right landscape position or not. We are actually made 2 different action, if the KTP is wrong orientation, it raise either error message or automatically rotate the KTP position until correct position. But finally, we are using auto rotate action algorithm to integrate with the apps.

---

We are using this notebook `auto_rotate.py`

### Features

- **Comparing white space** : This code indirectly learn the model of KTP images. The right KTP images should have more white pixels at right side.
- **Auto rotate image** : Check the orientation, if the width more than height (horizontal) then proceed to next step. If not, rotate image clockwise until raise the correct position. This code also implement grayscale, gaussian blur and thresolding for prepare comparing white pixels.

### Prerequisites

- OpenCV
- Numpy
- Matplotlib

