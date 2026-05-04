# Salient Object Detection

End-to-End ML/DL Project — Cohort V

## Dataset
DUTS-TR: 10,553 images with pixel-accurate saliency masks
Split: 70% Train / 15% Validation / 15% Test

## Model
Custom CNN Encoder-Decoder built from scratch (no pre-trained weights)
Encoder: 4x Conv2D + BatchNorm + ReLU + MaxPool
Decoder: 4x ConvTranspose2D + BatchNorm + ReLU + Skip Connections
Loss: BCE + 0.5 x (1 - IoU)
Optimizer: Adam lr=1e-3

## Results
| Model | IoU | Precision | Recall | F1 | MAE |
|-------|-----|-----------|--------|----|-----|
| Baseline | 0.8279 | 0.9206 | 0.9028 | 0.9116 | 0.0592 |
| Exp1: Dropout + Scheduler | 0.8306 | 0.9043 | 0.9218 | 0.9130 | 0.0599 |
| Exp2: IoU weight=1.0 | 0.8247 | 0.9064 | 0.9143 | 0.9103 | 0.0571 |

## Environment
- Python 3.9+
- PyTorch
- Google Colab GPU T4
