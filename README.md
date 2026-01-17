# Flowers-102 — Multi-Level Image Classification

[Gradio Space](https://huggingface.co/spaces/wrathog12/Flowers-102-ensemble)

A progressive deep learning pipeline for fine-grained image classification on the Oxford Flowers-102 dataset. This project demonstrates a staged approach to improving model performance, starting from a baseline transfer learning model and evolving into an attention-based ensemble architecture.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Methodology (Levels)](#methodology-levels)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
- [Results](#results)
- [Key Insights](#key-insights)
- [Contact](#contact)

---

## Overview

This repository addresses Fine-Grained Visual Categorization (FGVC) utilizing the Oxford Flowers-102 dataset. The solution is structured into four distinct levels of complexity, evaluating the impact of data augmentation, attention mechanisms, and ensemble strategies on classification accuracy.

## Problem Statement

The Oxford Flowers-102 dataset consists of 102 flower categories. The primary challenge is the high intra-class variance (differences in lighting, pose, and bloom stage within the same flower type) and low inter-class variance (subtle differences between similar species).

## Methodology (Levels)

The project is divided into four Jupyter notebooks, each representing a step up in complexity:

- **Level 1 — Baseline:**
  - Implements transfer learning using a pre-trained **ResNet-50** backbone.
  - Replaces the final fully connected layer to match 102 classes.
  - Uses standard normalization and resizing.

- **Level 2 — Augmentation & Fine-tuning:**
  - Introduces robust data augmentation: `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, and `ColorJitter`.
  - Implements selective fine-tuning by unfreezing Layer 3 and Layer 4 of the ResNet backbone to adapt high-level features to floral patterns.

- **Level 3 — Attention Mechanism (CBAM):**
  - Integrates **Convolutional Block Attention Modules (CBAM)** into the ResNet-50 architecture.
  - Applies both Channel and Spatial attention sequentially after every residual block to focus the model on relevant discriminative features (e.g., petals, textures) while suppressing background noise.

- **Level 4 — Ensemble Learning:**
  - Implements a **Soft-Voting Ensemble** strategy.
  - Combines the probability outputs of the "Augmented ResNet-50" (Level 2) and the "Attention-ResNet-50" (Level 3).
  - Demonstrates that combining diverse models yields better generalization than individual models.

## Repository Structure

```text
├── README.md              # Project documentation
├── Technical Report.pdf   # Detailed analysis, observations, and theoretical background
├── level1.ipynb           # Baseline model implementation
├── level2.ipynb           # Advanced data augmentation and fine-tuning experiments
├── level3.ipynb           # Custom Attention-ResNet-50 (CBAM) implementation
├── level4.ipynb           # Ensemble logic and final evaluation
└── requirements.txt       # Python dependencies
