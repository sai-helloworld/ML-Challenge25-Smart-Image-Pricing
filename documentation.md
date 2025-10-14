# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** [Absolut]  
**Team Members:** [Middela SaiPavan, Yerra Srikar Nivas Reddy, Sadha Shiva Aeshala]  
**Submission Date:** [13-10-2025]

---
## We are able to get 3505 out of 5523 submitted teams, and 3505 out of 20,000 registered teams
![WhatsApp Image 2025-10-14 at 19 00 05_c073e897](https://github.com/user-attachments/assets/c34a366a-1cb0-4050-97a6-a61e5a9f94db)



## 1. Executive Summary

_This solution employs a multimodal deep learning model that integrates image, text, and tabular data to predict product prices with high accuracy and resilience. Visual features are extracted using EfficientNet-B3 (pretrained, partially frozen), semantic signals are captured via TF-IDF vectorization of catalog content, and engineered tabular inputs include scaled numerical values and one-hot encoded units. These modalities are fused through a late fusion strategy and passed through fully connected layers for final regression.Training is optimized using mixed precision (AMP) and a custom SMAPE loss function to align tightly with competition metrics. To ensure robustness during long-running sessions, predictions are generated in 10,000-row chunks and saved incrementally. Additionally, model checkpoints are saved after each epoch, enabling seamless recovery and retraining in case of session interruptions on platforms like Kaggle or Google Colab._

---

## 2. Methodology Overview

### 2.1 Problem Analysis

_- Most of the rows have the product value and unit in the "catalog" column_
**Key Observations:**
-We have found that we can get Value and unit by using "Value:" and "Unit:" section at the end of text so We have extracted value and unit from catlog using regex of value: and unit: instead of going through all the text.

- We have found that there are no value and unit for some rows they are minimal so we have dropped them

### 2.2 Solution Strategy

**Approach Type:** Multimodal Single Model
**Core Innovation:** We propose a unified deep learning architecture that seamlessly fuses image, text, and tabular modalities to predict product prices with high interpretability and competitive SMAPE alignment. The model integrates:

Visual features via EfficientNet-B3 (pretrained on ImageNet, classifier removed, layers partially frozen)

Semantic cues via TF-IDF vectorization of catalog content (max_features=100)

Engineered tabular inputs including scaled numerical features and one-hot encoded units

These modalities are combined through late fusion, followed by fully connected layers for final regression output. Training is optimized using mixed precision (AMP) for speed and memory efficiency.

To ensure robustness and session resilience:

Chunked inference is employed—predicting 10,000 rows at a time and saving each chunk independently to avoid data loss during long-running sessions.

Checkpointing is implemented after every epoch, allowing seamless recovery and retraining from the last saved state in case of interruptions on platforms like Kaggle or Google Colab.

---

## 3. Model Architecture

### 3.1 Architecture Overview
## Text pipline
### catalog Text
<img width="348" height="326" alt="image" src="https://github.com/user-attachments/assets/fc0a3dd4-c4ce-468f-9abe-ff04df843499" />

### Derived Featueres

<img width="452" height="321" alt="image" src="https://github.com/user-attachments/assets/ab0cbf23-e891-43d1-b42e-0e3137875e2d" />


### Images pipline
<img width="483" height="636" alt="image" src="https://github.com/user-attachments/assets/c3ad458f-df9b-4fb9-853c-447fc90fd686" />


### 3.2 Model Components

**Text Processing Pipeline:**
Text Processing Pipeline
[x] Preprocessing Steps:

Lowercasing catalog_content

TF-IDF vectorization with max_features=100

[x] Model Architecture:

Fully connected layer: Linear → ReLU → Dropout

[x] Key Parameters:

Input dimension: 100

Hidden dimension: 64

Dropout rate: 0.2

**Tabular Processing Pipeline:**
[x] Preprocessing Steps:

One-hot encoding of unit_extracted (unknowns handled)

Standard scaling of value_extracted and num_of_packs

[x] Model Architecture:

Fully connected layer: Linear → ReLU → Dropout

[x] Key Parameters:

Input dimension: depends on encoded unit categories + 2 scaled features

Hidden dimension: 64

Dropout rate: 0.2

**Image Processing Pipeline:**
[x] Preprocessing Steps:

Resize to (300, 300)

Normalize using ImageNet mean and std

[x] Model Architecture:

EfficientNet-B3 (pretrained on ImageNet)

Classifier replaced with nn.Identity

[x] Key Parameters:

Frozen layers: features.0 to features.6

Output dimension: 1536

**Final Fusion Layer:**
[x] Concatenated Features:

Image (1536) + Tabular (64) + Text (64) → Total: 1664

[x] Architecture:

## Linear(1664 → 128) → ReLU → Linear(128 → 1)

## 4. Model Performance

### 4.1 Validation Results

- **SMAPE Score:** 0.5576

## 5. Conclusion

_This project successfully implemented a multimodal deep learning pipeline that fused image, text, and tabular features to predict product prices with high accuracy. By optimizing EfficientNet-B0, TF-IDF embeddings, and engineered tabular inputs under a unified architecture, the model achieved strong SMAPE performance while maintaining interpretability.Key lessons included the importance of robust preprocessing, modality balancing, and aligning loss functions with competition metrics for meaningful evaluation._

---

## Appendix

### A. Code artefacts

_[Include drive link for your complete code directory](https://drive.google.com/file/d/1TT8RRavmoAGACGvA5f_4_-V4TzUTV*GC/view?usp=sharing)*

### B. Additional Results

_Include any additional charts, graphs, or detailed results_
<img width="1086" height="679" alt="image" src="https://github.com/user-attachments/assets/cbb514e2-5c1d-4806-be82-2a6e5314a54f" />
<img width="1080" height="670" alt="image" src="https://github.com/user-attachments/assets/a5c95d0d-6b37-4f52-8614-ded02f4ad222" />
<img width="986" height="482" alt="image" src="https://github.com/user-attachments/assets/497d7e52-8d2c-42ce-b922-5dbded5329de" />


---
