# SERNet Image Segmentation API

This project is a REST API for image segmentation using a custom PyTorch model. The model is automatically downloaded from a GitHub release if not already present. The API accepts an input image and returns a colorized segmentation mask.

## Model Overview

- **Architecture**: SERNet (custom DeepLabV3-based)
- **Backbone**: ResNet-101
- **Framework**: PyTorch
- **Preprocessing**: Uses torchvision's `DeepLabV3_ResNet101_Weights.DEFAULT` transform
- **Performances**: Score IoU = 0.73


![frankfurt_000000_000294_leftImg8bit](https://github.com/user-attachments/assets/aa82e1e9-325e-4726-afd5-e503f6407da1)



The model is trained to segment images into 8 classes:  
  `void`, `flat`, `construction`, `object`, `nature`, `sky`, `human`, `vehicle`

## Features

- Downloads the trained model automatically from a GitHub release if not present
- Supports image segmentation with colored class masks
- Runs fully on CPU (no GPU required)
- Flask-powered REST API
- Logs detailed information and status
