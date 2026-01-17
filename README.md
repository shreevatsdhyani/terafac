# Terafac AI Hiring Challenge: Oxford Flowers-102 Classification

**Candidate Name:** Shreevats Dhyani
**Dataset Selected:** Option 3: Oxford Flowers-102
**Levels Completed:** Level 1, Level 2, Level 3, Level 4
**Highest Test Accuracy:** **90.55%** (Level 4 Ensemble)

---

## 📖 Executive Summary

This project tackles the fine-grained image classification task using the Oxford Flowers-102 dataset. The solution is structured into four progressive levels, evolving from baseline transfer learning to an advanced attention-based ensemble system.

**Key Achievements:**
*   **Robust Pipeline:** Implemented a reproducible PyTorch pipeline with seeded randomness.
*   **Advanced Architecture:** Designed a custom `AttentionResNet50` integrating **CBAM (Convolutional Block Attention Module)** for spatial and channel-wise feature refinement.
*   **Ensemble Strategy:** Achieved state-of-the-art performance using a Soft-Voting Ensemble of the augmented baseline and the attention-based model.
*   **Optimization:** Utilized differential learning rates and selective layer freezing to handle the small dataset size of Flowers-102 effectively.

---

## 📊 Performance Overview

| Level | Model Architecture | Method | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Level 1** | ResNet50 (Pretrained) | Baseline Transfer Learning | **85.2%** |
| **Level 2** | ResNet50 + Augmentation | Heavy Augmentation + Regularization | **88.13%** |
| **Level 3** | **Attention-ResNet50** | Custom CBAM Attention Modules | **88.83%** |
| **Level 4** | **Ensemble (L2 + L3)** | Soft-Voting Strategy | **90.55%** |

---

## 📂 Repository Structure

The project is organized into modular Google Colab notebooks for reproducibility.

*   **[Level 1 Notebook](https://drive.google.com/file/d/1yhBEmErBdwlp6leXrtWEOXApgi9QYvCR/view?usp=sharing)**: Baseline implementation.
*   **[Level 2 Notebook](https://drive.google.com/file/d/1bj_2pULuXdoZIMMZC2fWpKFJlSlZVKeK/view?usp=drive_link)**: Advanced data augmentation techniques.
*   **[Level 3 Notebook](https://drive.google.com/file/d/1eA8A9NhbrWEpYffJs3JcBxjRzZ0zRFoe/view?usp=drive_link)**: Custom Architecture design (CBAM).
*   **[Level 4 Notebook](https://drive.google.com/file/d/16we_9dPeowoxuiKRnxzPCwN6fOPq7DdS/view?usp=drive_link)**: Ensemble modeling and final evaluation.

---

## 🛠️ Level-wise Methodology

### Level 1: Baseline Model
**Goal:** Establish a performance baseline using Transfer Learning.
*   **Architecture:** ResNet50 (ImageNet weights).
*   **Data Split:** 80% Train / 10% Val / 10% Test (Strict adherence to challenge rules).
*   **Preprocessing:** Standard Resize (224x224) and ImageNet Normalization.
*   **Observation:** The model achieved respectable accuracy (~85%) but showed signs of overfitting due to the small size of the Flowers-102 dataset.

### Level 2: Intermediate Techniques (Data Augmentation)
**Goal:** Combat overfitting and improve generalization.
*   **Techniques Used:**
    *   `RandomResizedCrop`: Forces the model to learn local features.
    *   `RandomHorizontalFlip`: Geometric invariance.
    *   `RandomRotation`: Orientation invariance.
    *   `ColorJitter`: Robustness to lighting conditions.
*   **Training Strategy:** Selective fine-tuning. We unfroze layers 3 and 4 of the ResNet backbone to allow adaptation to floral features while keeping early texture-detection layers frozen.
*   **Result:** Accuracy improved to **88.13%**, confirming that data diversity was the primary bottleneck.

### Level 3: Advanced Architecture (Attention Mechanisms)
**Goal:** Design a custom architecture to focus on fine-grained details (petals, textures).
*   **Innovation:** Integrated **CBAM (Convolutional Block Attention Module)**.
*   **Implementation:**
    *   Created `AttentionResNet50` class.
    *   Injected CBAM blocks after every ResNet bottleneck layer (`layer1` through `layer4`).
    *   **Channel Attention:** "What" features are important?
    *   **Spatial Attention:** "Where" are the features located?
*   **Result:** Accuracy reached **88.83%**. The model demonstrated better localization of flower boundaries compared to the standard ResNet.

### Level 4: Expert Techniques (Ensemble Learning)
**Goal:** Push performance boundaries by combining diverse models.
*   **Strategy:** **Soft-Voting Ensemble**.
    *   Model A: `Level 2 ResNet50` (Strong generalization via augmentation).
    *   Model B: `Level 3 AttentionResNet50` (Strong feature localization via CBAM).
*   **Inference Logic:**
    1.  Pass image through Model A $\rightarrow$ Softmax Probabilities $P_A$.
    2.  Pass image through Model B $\rightarrow$ Softmax Probabilities $P_B$.
    3.  Compute Weighted Average: $P_{final} = \alpha P_A + (1-\alpha) P_B$ (where $\alpha=0.5$).
    4.  Argmax for final prediction.
*   **Outcome:** The ensemble achieved **90.55%** accuracy, surpassing both individual models by correcting distinct error modes unique to each architecture.

---

## 📈 Visualizations and Analysis

### 1. Training Dynamics
*Note: Please refer to the notebooks for the generated Loss/Accuracy plots.*
The training curves indicate that the Level 2 and Level 3 models converge quickly. The validation loss stabilized around Epoch 5-7, suggesting that further training without stronger regularization (like Mixup/Cutmix) would yield diminishing returns.

### 2. Failure Analysis
Upon analyzing the confusion matrix, specific misclassifications persist among visually similar classes (e.g., *'passion flower'* vs *'lotus'*).
*   **Reasoning:** These classes differ only by subtle pistil structures which are lost at 224x224 resolution.
*   **Solution for Level 5:** Increasing resolution to 448x448 or using Vision Transformers (ViT) would likely resolve these fine-grained ambiguities.

---

## 📦 Requirements

To run the notebooks, the following dependencies are required (installed automatically in Colab):

```text
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
tensorflow_datasets (for data download)
tqdm
```

---

## 📝 Final Conclusion

This project demonstrates that while standard transfer learning is powerful, fine-grained tasks like Flower classification require specific architectural interventions. The introduction of **CBAM attention** helped the model focus on relevant floral features, and **Ensembling** effectively reduced variance, leading to a production-grade accuracy of **90.55%**.

---
*Submitted as part of the Terafac AI Hiring Challenge.*
