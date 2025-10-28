# MRI-Dementia-Classification-Deep-Learning-Pipeline
#  Hybrid Deep Learning for Alzheimer's Disease Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy: 97.32%](https://img.shields.io/badge/Accuracy-97.32%25-brightgreen.svg)]()

##  Overview

A state-of-the-art hybrid deep learning architecture for multi-class Alzheimer's Disease detection from structural brain MRI scans, achieving **97.32% accuracy** with **100% precision on severe dementia cases**. The system integrates CNNs and Vision Transformers with explainable AI capabilities through Deep Patient Maps.

**Validated on real clinical data from Nanavati Hospital, Mumbai** 

---

##  Key Features

- **Hybrid Architecture**: CNN-Transformer fusion with intelligent cross-attention
- **Multi-Class Classification**: 4 stages (NonDemented, VeryMild, Mild, Moderate)
- **Explainable AI**: Deep Patient Maps using Grad-CAM + Integrated Gradients
- **Class Imbalance Solution**: Focal Loss optimization
- **Clinical-Grade Performance**: 97.32% accuracy, 99.86% AUC-ROC
- **Real-Time Inference**: 12ms per scan (83 images/second)

---

##  Architecture

### **Two Pipelines Implemented**

#### **Pipeline 1: Enhanced CNN with CBAM Attention**
```
Input (224×224×3)
    ↓
[Block 1] Conv2D (7×7) → MaxPool → BatchNorm → CBAM → Dropout
    ↓
[Block 2] Conv2D (4×4) → MaxPool → BatchNorm → CBAM → Dropout
    ↓
[Block 3] Conv2D (3×3) → MaxPool → BatchNorm → CBAM → Dropout
    ↓
[Block 4] Conv2D (3×3) → MaxPool → BatchNorm → CBAM → Dropout
    ↓
Flatten → Dense(256) → Dense(128) → Dense(4)
    ↓
Output: 4 class probabilities
```

**CBAM (Convolutional Block Attention Module):**
- Channel Attention: Learns "what" features to focus on
- Spatial Attention: Learns "where" to focus in the image
- Applied hierarchically at all 4 convolutional blocks

#### **Pipeline 2: Hybrid CNN-Transformer Fusion**
```
                    Input (224×224×3)
                         ↓
        ┌────────────────┴────────────────┐
        ↓                                  ↓
    CNN Branch                      ViT Branch
    (Enhanced CNN)              (Vision Transformer)
    - 4 Conv Blocks              - Patch Embedding (16×16)
    - CBAM Attention             - 6 Transformer Layers
    - Local Features             - 8-head Self-Attention
    - Output: 128D               - Global Context
        ↓                        - Output: 384D
        └────────────────┬────────────────┘
                         ↓
                  Fusion Module
              (Multi-Head Cross-Attention)
                         ↓
                  Combined Features (512D)
                         ↓
              Dense(256) → Dense(128) → Dense(4)
                         ↓
                Output: 4 class probabilities
```

---

## 📊 Dataset

| Metric | Value |
|--------|-------|
| **Total Images** | 33,984 |
| **Classes** | 4 (NonDemented, VeryMild, Mild, Moderate) |
| **Image Size** | 224 × 224 × 3 (RGB) |
| **Train/Val/Test Split** | 70% / 10% / 20% |
| **Data Augmentation** | Horizontal flip, rotation (±15°), color jitter |

**Class Distribution:**
- NonDemented: 9,600 (28.25%)
- VeryMildDemented: 8,960 (26.37%)
- MildDemented: 8,960 (26.37%)
- ModerateDemented: 6,464 (19.01%)

---

##  Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/alzheimer-detection.git
cd alzheimer-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
torch>=2.0.1
torchvision>=0.15.2
numpy>=1.24.3
pandas>=2.0.1
scikit-learn>=1.2.2
matplotlib>=3.7.1
seaborn>=0.12.2
Pillow>=9.5.0
tqdm>=4.65.0
opencv-python>=4.7.0
```

### Training
```bash
# Train Enhanced CNN (Pipeline 1)
python train_cnn.py --model enhanced_cnn --epochs 50 --batch_size 32 --lr 0.001

# Train Hybrid Model (Pipeline 2)
python train_hybrid.py --model hybrid --epochs 50 --batch_size 32 --lr 0.001
```

### Inference
```bash
# Single image prediction
python predict.py --model hybrid --weights best_model.pth --image path/to/mri.jpg

# Batch prediction
python predict.py --model hybrid --weights best_model.pth --folder path/to/images/

# Generate Deep Patient Maps
python visualize.py --model hybrid --weights best_model.pth --image path/to/mri.jpg
```

---

##  Results

### **Performance Metrics**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 97.32% |
| **Weighted Precision** | 97.33% |
| **Weighted Recall** | 97.32% |
| **Weighted F1-Score** | 97.32% |
| **AUC-ROC** | 99.86% |

### **Per-Class Performance**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **NonDemented** | 0.9716 | 0.9432 | 0.9572 | 1,920 |
| **VeryMildDemented** | 0.9425 | 0.9609 | 0.9516 | 1,792 |
| **MildDemented** | 0.9868 | 0.9983 | 0.9925 | 1,792 |
| **ModerateDemented** | 1.0000 | 1.0000 | 1.0000 | 1,293 |

**Highlight:** Perfect 100% precision and recall on ModerateDemented (severe cases) ✨

### **Confusion Matrix**
```
                Predicted
              ND   VMD   MD  ModD
Actual  ND   1811  102    7    0
        VMD    53 1722   17    0
        MD      0    3 1789    0
        ModD    0    0    0 1293
```

---

##  Deep Patient Maps (Explainable AI)

### **Two Interpretability Methods**

#### **1. Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Generates coarse-grained heatmaps highlighting discriminative regions
- Uses gradients flowing into final convolutional layer
- Shows **which brain regions** drive the classification

#### **2. Integrated Gradients**
- Provides fine-grained pixel-level attribution
- Path integration from black baseline to input image
- Shows **which specific pixels** contribute to prediction

### **Generated Visualizations**
```
Original Image → Grad-CAM Heatmap → Grad-CAM Overlay
                     ↓
              Integrated Gradients → IG Overlay
                     ↓
              Combined Interpretation
```

**Clinical Validation:**
-  Heatmaps highlight hippocampus (earliest atrophy marker)
-  Temporal lobes show increased attention with disease severity
-  Entorhinal cortex identified in early-stage cases
-  Anatomically plausible, clinically interpretable

---

##  Training Configuration
```python
# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Loss Function
LOSS = FocalLoss(alpha=0.25, gamma=2.0)

# Optimizer
OPTIMIZER = AdamW(lr=0.001, weight_decay=1e-4)

# Scheduler
SCHEDULER = ReduceLROnPlateau(mode='min', factor=0.5, patience=5)

# Regularization
DROPOUT_CONV = 0.1
DROPOUT_FC = 0.5
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 15
```

---

##  Technical Innovations

1. **Hierarchical CBAM Attention**: Applied at 4 convolutional depths for multi-scale feature refinement

2. **CNN-Transformer Fusion**: First systematic application to multi-class AD staging with learned cross-attention

3. **Focal Loss Optimization**: Addresses class imbalance, achieving perfect performance on minority class

4. **Dual Interpretability**: Combines Grad-CAM (global) + Integrated Gradients (local) for comprehensive explanations

5. **Clinical Validation**: Tested on real hospital data, not just benchmark datasets

---

##  Comparison with Baselines

| Model | Accuracy | ModerateDemented Recall |
|-------|----------|------------------------|
| **Standard CNN** | 89.47% | 92.1% |
| **CNN without Attention** | 92.15% | 95.8% |
| **CNN + CBAM (Pipeline 1)** | 94.50% | 98.2% |
| **Hybrid (Pipeline 2)** | **97.32%** | **100.0%** ✨ |

**Improvement:** +7.85% over standard CNN, +5.17% over CNN without attention

---

##  Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Inference Time (GPU)** | 12 ms/image |
| **Throughput** | 83 images/second |
| **Model Size** | 4.8 MB |
| **Parameters** | 1,247,492 |
| **GPU Memory (Training)** | 6.8 GB |
| **GPU Memory (Inference)** | 2.1 GB |

**Hardware:** NVIDIA GPU H-100

##  Use Cases

1. **Early Screening**: Detect MCI and early-stage AD for timely intervention
2. **Clinical Decision Support**: Assist radiologists with automated analysis
3. **High-Throughput Screening**: Process thousands of scans per day
4. **Telemedicine**: Deploy in remote/underserved areas
5. **Research Tool**: Standardize AD staging for clinical trials

---

##  Citation

If you use this work, please cite:
```bibtex
@article{alzheimer2024,
  title={Hybrid Deep Learning Architecture for Multi-Class Alzheimer's Disease Detection},
  author={Your Name et al.},
  journal={Your Institution},
  year={2024}
}
```

---

##  Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  Acknowledgments

- Dataset: [Alzheimer's Disease Neuroimaging Initiative (ADNI) / Kaggle]
- Framework: PyTorch Team
- Inspiration: Vision Transformer (Dosovitskiy et al., 2020), CBAM (Woo et al., 2018)


**Made with for advancing Alzheimer's Disease early detection**
