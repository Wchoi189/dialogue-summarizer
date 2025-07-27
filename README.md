# Dialogue Summarization with PyTorch Lightning & Hydra

A modular, production-ready dialogue summarization system built with PyTorch Lightning and Hydra configuration management.

## 🚀 Quick Start

1.  **Create Environment**
    ```bash
    # Create a minimal environment with Python
    micromamba env create -f environment.yml
    micromamba activate dialogue-summarization
    ```

2.  **Install Dependencies**
    ```bash
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

    # Install remaining packages
    pip install -r requirements.txt
    ```

3.  **Train Model**
    ```bash
    python scripts/train.py train --config-name config
    ```

4.  **Generate Predictions**
    ```bash
    python scripts/inference.py submission /path/to/best/model.ckpt
    ```


```markdown
## 📁 Project Structure

```

dialogue\_summarization/
├── configs/          \# Hydra configuration files
├── src/              \# Source code
│   ├── data/         \# Data processing
│   ├── models/       \# Model implementations
│   ├── evaluation/   \# Evaluation metrics
│   ├── inference/    \# Inference pipeline
│   └── utils/        \# Utilities
├── scripts/          \# Entry point scripts
├── notebooks/        \# Jupyter notebooks
└── tests/            \# Unit tests

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
