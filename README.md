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

    # Use minimal postprocessing for debugging
    python scripts/train.py train --config-name config --override postprocessing=minimal

    ```

    **Debug training run**
    ```bash
    python scripts/train.py train --override training=production
    python scripts/train.py train --override experiment=production
    python scripts/train.py train --experiment production --override
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
    '/home/wb2x/workspace/dialogue-summarizer/outputs/models/best-epoch=06-val/rouge_f=0.4226.ckpt' \
    --output-file submission1.csv
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

```markdown

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
```

```markdown
That's an excellent job running the commands. The output is very revealing and points to a single, critical error in your configuration that has caused the massive 59GB of storage usage.

The root cause is a forward slash (`/`) in your checkpoint filename. Your system is interpreting it as a directory separator, creating a new 9.7GB directory for almost every checkpoint instead of a single 1.4GB file.

-----

### The Root Cause: A Slash in the Filename

In your `scripts/train.py` file, the `ModelCheckpoint` is configured with this filename: `"best-{epoch:02d}-{val/rouge_f:.4f}"`.

The `{val/rouge_f}` part is the problem. The `/` causes the system to create a directory named `best-epoch=02-val` and then save the checkpoint inside it. This has been happening repeatedly, filling up your disk.

-----

### Step 1: Fix the Code (Permanent Solution)

First, let's fix the filename to prevent this from ever happening again.

  * **File to Edit**: `scripts/train.py`
  * **Action**: In the `_setup_trainer` method, change the `/` in the `ModelCheckpoint` filename to a more filename-friendly character like `=` or `_`.

<!-- end list -->

```python
# In scripts/train.py -> _setup_trainer()

# --- BEFORE ---
# checkpoint_callback = ModelCheckpoint(
#     dirpath=output_dir / "models",
#     filename="best-{epoch:02d}-{val/rouge_f:.4f}",
#     ...
# )

# --- AFTER ---
checkpoint_callback = ModelCheckpoint(
    dirpath=output_dir / "models",
    filename="best-{epoch:02d}-val_rouge_f={val/rouge_f:.4f}", # ✅ Corrected filename
    monitor="val/rouge_f",
    mode="max",
    save_top_k=1, # Also recommend setting this to 1 to save only the best
    save_last=True,
    verbose=True,
)
```

-----

### Step 2: Clean Up the Existing Training Files

Now, let's reclaim your disk space. Your `find` command shows that your best-performing checkpoint so far is `best-epoch=03-val/rouge_f=0.4376.ckpt`. We will save that single file and delete everything else.

Run these commands from your `dialogue-summarizer` directory:

**1. Create a safe temporary directory**

```bash
mkdir -p outputs/safe_checkpoint
```

**2. Move your single best checkpoint to safety**

```bash
mv outputs/models/best-epoch=03-val/rouge_f=0.4376.ckpt outputs/safe_checkpoint/
```

**3. Delete the entire 59GB `models` directory**

> **WARNING**: This command will permanently delete the `outputs/models` directory and all of its contents. Double-check that you have moved your best checkpoint first.

```bash
rm -rf outputs/models
```

**4. Recreate the directory and move the best checkpoint back**

```bash
mkdir outputs/models
mv outputs/safe_checkpoint/rouge_f=0.4376.ckpt outputs/models/
rm -r outputs/safe_checkpoint
```
## Clear PyCache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

```