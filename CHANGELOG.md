# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-05-22

### Added

**Data Pipeline (`src/data_pipeline_optimized.ipynb`)**
- Frame extraction from raw MP4 videos at 1 FPS (max 30 frames per video)
- Batched face detection and cropping using MTCNN (224×224 px)
- Video-level train/val/test split (70/15/15) to prevent data leakage
- Resume support — skips already-processed frames on reconnection
- Fix: replaced `face_detector(batch)` with `face_detector.detect()` +
  `extract_face()` to resolve numpy inhomogeneous-array crash when some
  frames contain no detectable face
- Fix: added `--force-reinstall Pillow` after facenet-pytorch install to
  resolve `PIL._util.is_directory` ImportError on Google Colab

**Model Training (`src/model_training_evaluation.ipynb`)**
- EfficientNet-B0 with ImageNet pretrained weights
- Two-phase transfer learning:
  - Experiment 1: frozen backbone, LR = 1e-3 → val acc 88.75%
  - Experiment 2: fine-tune last 2 blocks, LR = 1e-4 → val acc 93.86%
- Adam optimizer with weight decay (1e-4), ReduceLROnPlateau scheduler
- Full test-set evaluation: accuracy, confusion matrix, ROC/AUC
- Video-level majority-vote aggregation
- Grad-CAM visualisations for real and fake faces

**Results**
- Frame-level accuracy: **93.27%**
- Video-level accuracy: **97.77%**
- AUC: **0.9573**
- Dataset: 82,052 face images from DFD (363 real / 3,068 fake videos)

**Repository**
- Organised into `src/`, `models/`, `results/`, `configs/`, `docs/`
- README with results tables, architecture overview, and setup guide
- MIT License, CONTRIBUTING guide, GitHub Issue templates
