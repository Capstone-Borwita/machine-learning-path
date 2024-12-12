# Images for Segmentation Test

There are some example of images spesifically for segmentation test. Noted there are some requirement of images so that it can pass the model. 
- `fitKTP.jpg` shouldn't pass because it touch the image border. Altohough it is full ktp image, 
- `fotokopiKTP.jpg` should pass because it is whole KTP and all the sides doesn't touch the image border.
- `fotokopiKTP2.jpg` shouldn't pass because there it touch image border (the image edges almost unseen)
- `halfKTP.png` shouldn't pass because it touch the image border.
- `onHandKTP.jpg` should pass because it still good KTP photo, then the SegmentationAndCrop code will be proceed into nice cropped KTP Images.
- `perfectScenario.jpg` should pass because it very good KTP photo.
- `wholeKTP.jpg` should pass because it still a nice KTP photo.
