# Dialogue Summarization with PyTorch Lightning & Hydra

A modular, production-ready dialogue summarization system built with PyTorch Lightning and Hydra configuration management.

## 🚀 Quick Start

1.  **Create Environment**
    ```bash
    # Create a minimal environment with Python
    micromamba env create -f environment.yml
    micromamba activate dialogue-summarization
    ```
    **Remove Environment**
    ```bash
    micromamba env list
    rm -rf /opt/conda/envs/dialogue-summarization
    or
    micromamba remove -n dialogue-summarization --all
    ```
2.  **Install Dependencies**
    ```bash
    # Install PyTorch with CUDA support
    pip3 install --pre torch==2.6.0.dev20241112+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --no-cache-dir
    # Install remaining packages
    pip install -r requirements.txt
    ```

3.  **Train Model**
    ```bash
    python scripts/train.py train --config-name config
    python scripts/train.py train --config-name kobart-base-v2
    python scripts/train.py train --config-name config
    ```

    **Debug training run**
    ```bash
    python scripts/train.py train --override training=baseline-debug
    ```
3.  **Custom Postprocesssing**
    ```bash
    # Use default postprocessing
    python scripts/train.py train --config-name config 

    # Use aggressive postprocessing
    python scripts/train.py train --config-name config --override postprocessing=aggressive
    
    # Use minimal postprocessing for debugging
    python scripts/train.py train --config-name config --override postprocessing=minimal

    # Use custom postprocessing settings via command line
    python scripts/train.py train --config-name config --override postprocessing.remove_tokens=["<usr>","<pad>"] postprocessing.text_cleaning.strip_whitespace=true

    # For inference
    python scripts/inference.py submission /path/to/model.ckpt --override postprocessing=aggressive
    python scripts/inference.py submission /path/to/best/model.ckpt --override postprocessing=aggressive
    ```

4.  **Generate Predictions**
    ```bash
    python scripts/inference.py submission /path/to/best/model.ckpt

    python scripts/inference.py submission \
    /home/wb2x/workspace/dialogue-summarizer/outputs/models/last.ckpt \
    --output-file submission.csv
    ```
**Wandb login**
```bash
export WANDB_API_KEY="YOUR_API_KEY"
source ~/.bashrc
wandb login
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
