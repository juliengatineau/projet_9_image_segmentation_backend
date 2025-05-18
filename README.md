# SERNet Image Segmentation API

This project is a REST API for image segmentation using a custom PyTorch model. The model is automatically downloaded from a GitHub release if not already present. The API accepts an input image and returns a colorized segmentation mask.

## Model Overview

- **Architecture**: SERNet (custom DeepLabV3-based)
- **Backbone**: ResNet-101
- **Framework**: PyTorch
- **Preprocessing**: Uses torchvision's `DeepLabV3_ResNet101_Weights.DEFAULT` transform
- **Performances**: Score IoU = 0.73

  Real mask
  ![frankfurt_000000_000294_leftImg8bit](https://github.com/user-attachments/assets/04d0ead2-dc7a-446d-bd21-8a8ef477752c)

  Predicted mask
  ![frankfurt_000000_000294_leftImg8bit_sernet_mask](https://github.com/user-attachments/assets/3b4d9960-55c6-49e7-8735-618d6f1c3f41)

The model is trained to segment images into 8 classes:  
  `void`, `flat`, `construction`, `object`, `nature`, `sky`, `human`, `vehicle`

## Features

- Downloads the trained model automatically from a GitHub release if not present
- Supports image segmentation with colored class masks
- Runs fully on CPU (no GPU required)
- Flask-powered REST API
- Logs detailed information and status
