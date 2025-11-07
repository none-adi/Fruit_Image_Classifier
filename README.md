# 🍎 Multi-Fruit Ripeness Classification: A Data-Efficient Benchmark of CNNs and Vision Transformers

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧠 Project Overview

This repository presents a **comparative benchmarking study** of modern deep learning architectures for **multi-fruit ripeness classification**.  
The goal is to evaluate and analyze the **data efficiency, generalization, and performance trade-offs** of different Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) on a public fruit dataset — without creating any new data.

We benchmark four pretrained architectures:
- 🪶 **MobileNetV2** – Lightweight CNN optimized for edge devices  
- ⚙️ **EfficientNet-B0** – State-of-the-art CNN balancing accuracy and efficiency  
- 🧩 **ResNet50** – Classic deep CNN baseline  
- 🔭 **ViT-B/16** – Vision Transformer leveraging global self-attention  

---

## 🍇 Dataset

- **Name:** [Fruit Image Dataset: 22 Classes (Kaggle)](https://www.kaggle.com/datasets)
- **Composition:** 22 fruit–ripeness categories (e.g., `ripe_apple`, `unripe_apple`, `ripe_banana`, etc.)
- **Split:** 70% Train | 15% Validation | 15% Test  
- **Total Samples:** ~18,000 images  
- **Image Size:** Resized to 224×224  
- **Augmentations:**
  - RandomResizedCrop, Rotation, HorizontalFlip
  - ColorJitter (brightness, contrast, saturation, hue)
  - Normalization (ImageNet mean/std)

---

## ⚙️ Methodology

Each model was trained under **identical experimental settings** for fairness:

| Parameter | Value |
|:--|:--|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.05 |
| Loss Function | CrossEntropy with Label Smoothing (0.1) |
| Batch Size | 32 |
| Epochs | 50 |
| Framework | PyTorch |
| Hardware | Kaggle GPU (NVIDIA T4) |

---

## 🧩 Training Pipeline

```text
Dataset → Preprocessing → Model Initialization → Training Loop
          → Validation → Metrics Logging → Visualization → Model Saving

Dataset Used: https://www.kaggle.com/datasets/mdsagorahmed/fruit-image-dataset-22-classes
