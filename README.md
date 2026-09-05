---
title: Bacterial Colony Classifier
emoji: 🧫
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.26.0
python_version: 3.11
app_file: app.py
pinned: false
---

# Bacterial Colony Classifier

A deep learning application for classifying bacterial colony images into 19 bacterial species using ResNet-18.

## Models

The application provides two trained ResNet-18 models:

- **Model 1:** Experiment A — unseen-strain split
- **Model 2:** Experiment B — random image-level split

Users can select either model and upload a bacterial colony image to obtain the predicted species and confidence score.

## Input

The model accepts bacterial colony images and applies the same preprocessing used during model development:

- Resize with padding to 224 × 224
- Convert to RGB
- Convert to tensor
- ImageNet normalization

## Output

The application displays:

- Predicted bacterial species
- Prediction confidence

## Disclaimer

This application is intended for research and educational purposes and should not be used as a substitute for laboratory identification or clinical diagnosis.