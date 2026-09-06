# Bacterial Colony Classifier

A deep learning application for classifying high-resolution bacterial colony images into 19 bacterial species using ResNet-18.

## **Models**

Two ResNet-18 classification models are provided:

* **Model 1:** Experiment A — unseen-strain split
* **Model 2:** Experiment B — random image-level split

Both models classify complete plate images without colony bounding-box cropping.

## **Preprocessing**

Input images are processed using the same pipeline used during model development:

* Resize with padding to 224 × 224 pixels
* Convert to RGB
* Convert to tensor
* ImageNet normalization

## **Prediction**

The application allows users to:

* Upload a bacterial colony image
* Select Model 1 or Model 2
* Obtain the predicted bacterial species
* View the prediction confidence

## **Technologies**

Python, PyTorch, Torchvision, ResNet-18, Streamlit, Pillow

## **Deployment**

The application is designed for deployment using Streamlit Community Cloud.

## **Disclaimer**

This application is intended for research and educational purposes only. It should not be used as a substitute for laboratory identification or clinical diagnosis.

