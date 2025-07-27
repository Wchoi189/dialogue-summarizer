# Dialogue Summarization with PyTorch Lightning & Hydra

A modular, production-ready dialogue summarization system built with PyTorch Lightning and Hydra configuration management.

## 🚀 Quick Start

1. **Environment Setup**
   ```bash
   micromamba env create -f environment.yml
   micromamba activate dialogue-summarization
   ```

2. **Initialize Project**
   ```bash
   python scripts/setup_project.py
   ```

3. **Train Model**
   ```bash
   python scripts/train.py
   ```

4. **Generate Predictions**
   ```bash
   python scripts/inference.py
   ```

## 📁 Project Structure

```
dialogue_summarization/
├── configs/          # Hydra configuration files
├── src/             # Source code
│   ├── data/        # Data processing
│   ├── models/      # Model implementations  
│   ├── training/    # Training logic
│   ├── evaluation/  # Evaluation metrics
│   ├── inference/   # Inference pipeline
│   └── utils/       # Utilities
├── scripts/         # Entry point scripts
├── notebooks/       # Jupyter notebooks
└── tests/          # Unit tests
```

## 🛠️ Configuration

This project uses Hydra for configuration management. Key config groups:

- `model`: Model architecture (kobart, solar_api)
- `dataset`: Data processing settings
- `training`: Training hyperparameters
- `inference`: Generation parameters

## 📊 Performance

Current baseline: **ROUGE-F1: 47.1244**

## 🔬 Models

- **KoBART**: Korean BART model for dialogue summarization
- **Solar API**: Alternative approach using Solar Chat API

## 📈 Experiment Tracking

Integration with Weights & Biases for experiment tracking and model versioning.
