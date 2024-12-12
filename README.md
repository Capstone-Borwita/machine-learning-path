# Machine Learning Overview

This repository contains code and files for KTP Data Extraction. We implementing OCR (Optical Character Recognition) method to extract the data, but there are also some essensial code to make the OCR process run smoothly and raise the best accuracy.

## Dataset
The dataset contains approx 350++ KTP images for training all the model. And there are also cropped NIK, Name, Address images for train the OCR. Sometimes we also use sythetic data that has been made as similar as possible with the image quality (especially for cropped images) to improve the accuracy. For addition, there are also Non KTP images data for trained the image classification.
You can find the data [here](https://drive.google.com/drive/u/0/folders/1ng6b2Lx0fhWm86gHl-paGIuFpg25xjdD) (PLEASE USE IT WISELY!!)

## Contents
This repository features the following files :
- **KTP_Classifier** : Contain files for detect the images wheter it's KTP or non-KTP. Include code for training and testing using Keras.
- **SegmentationAndCrop** : This folder provides YOLO-based tools for detecting, segmenting, cropping, validating, and transforming Indonesian ID cards (KTP).
- **readjust_lanscape** : There is code to make sure KTP orientation and auto-rotate if it's not at correct position.
- **Object_Detection_Bounding_Box** : This folder extends KTP detection and segmentation by using YOLOv8 to label and crop fields (NIK, NAMA, Alamat) for OCR processing.
- **OCR** :  This folder contains CNN-RNN based OCR code (both training and testing) to generate text based on image text. It works on 2 different model font for NIK and for Name, Address.
- **batch-test** : Including OCR performance test and comparison to other OCR model that had existed before.
- **model** : Set of model that will be used on application.
- **testImages** : KTP images for various scenarios.
- **Utils** : Set of code for deployment purpose.
- **api.py** : File for integrated with endpoint API.
- **main.py** : Function for testing script locally without needing to go through an API endpoint.
