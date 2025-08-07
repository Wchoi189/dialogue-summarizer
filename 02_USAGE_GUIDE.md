# =============================================================================
# COMMAND LINE USAGE EXAMPLES - After train.py modification
# =============================================================================

# 1. BASIC EXPERIMENT USAGE
# Use baseline experiment (your current setup)
python scripts/train.py train --experiment baseline

# Use aggressive experiment (higher learning rate, longer training)
python scripts/train.py train --experiment aggressive

# Use debug experiment (fast, small batches)
python scripts/train.py train --experiment debug

# Use production experiment (optimized for final model)
python scripts/train.py train --experiment production

# =============================================================================
# 2. OVERRIDING SPECIFIC PARAMETERS
# =============================================================================

# Override max_epochs using the new dedicated flag
python scripts/train.py train --experiment baseline --max-epochs 12

# Override batch_size using the new dedicated flag
python scripts/train.py train --experiment baseline --batch-size 32

# Override learning_rate using the --override syntax
python scripts/train.py train --experiment baseline --override training.optimizer.lr=1e-5

# Combine multiple overrides
python scripts/train.py train \
    --experiment baseline \
    --max-epochs 10 \
    --batch-size 24 \
    --override training.optimizer.lr=8e-6

# =============================================================================
# 3. ADVANCED CONFIG OVERRIDES (Using --override)
# =============================================================================

# Override generation parameters
python scripts/train.py train \
    --experiment baseline \
    --override generation.max_length=80 \
    --override generation.repetition_penalty=1.4

# Override model settings
python scripts/train.py train \
    --experiment baseline \
    --override model.compile.enabled=false

# Override wandb settings
python scripts/train.py train \
    --experiment baseline \
    --override wandb.offline=true \
    --override wandb.tags=[test,experiment]

# Override multiple data settings
python scripts/train.py train \
    --experiment baseline \
    --override data.batch_size=64 \
    --override data.num_workers=16 \
    --override data.pin_memory=false

# =============================================================================
# 4. DEBUGGING AND DEVELOPMENT
# =============================================================================

# Quick debug run (1 batch)
python scripts/train.py train --experiment debug --fast-dev-run

# Debug with specific settings
python scripts/train.py train \
    --experiment debug \
    --max-epochs 1 \
    --batch-size 2 \
    --override wandb.offline=true

# Test config loading without training
python scripts/train.py train \
    --experiment baseline \
    --override training.fast_dev_run=true \
    --override training.limit_train_batches=0.01

# =============================================================================
# 5. VALIDATION COMMANDS
# =============================================================================

# Validate with specific experiment config
python scripts/train.py validate \
    --experiment baseline \
    --checkpoint-path outputs/models/best-epoch=05-val_rouge_f=0.1234.ckpt

# Validate with overrides
python scripts/train.py validate \
    --experiment baseline \
    --checkpoint-path path/to/model.ckpt \
    --override generation.num_beams=6

# =============================================================================
# 6. RESUME TRAINING
# =============================================================================

# Resume from checkpoint with same experiment
python scripts/train.py train \
    --experiment baseline \
    --resume-from outputs/models/last.ckpt

# Resume but change some settings
python scripts/train.py train \
    --experiment baseline \
    --resume-from outputs/models/last.ckpt \
    --override training.optimizer.lr=3e-6 \
    --override training.early_stopping.patience=1

# =============================================================================
# 7. PRODUCTION TRAINING EXAMPLES
# =============================================================================

# Production training with optimal settings
python scripts/train.py train \
    --experiment production \
    --max-epochs 20 \
    --override wandb.tags=[production,final,v1]

# Custom production run
python scripts/train.py train \
    --experiment production \
    --batch-size 64 \
    --learning-rate 2e-6 \
    --override generation.max_length=45 \
    --override generation.num_beams=5 \
    --override training.early_stopping.patience=5

# =============================================================================
# 8. COMPARISON EXPERIMENTS
# =============================================================================

# Compare different generation lengths
python scripts/train.py train \
    --experiment baseline \
    --override generation.max_length=40 \
    --override experiment_name=baseline_short_summaries

python scripts/train.py train \
    --experiment baseline \
    --override generation.max_length=80 \
    --override experiment_name=baseline_long_summaries

# Compare different learning rates
python scripts/train.py train \
    --experiment baseline \
    --learning-rate 3e-6 \
    --override experiment_name=baseline_lr3e6

python scripts/train.py train \
    --experiment baseline \
    --learning-rate 1e-5 \
    --override experiment_name=baseline_lr1e5

# =============================================================================
# 9. ERROR CHECKING EXAMPLES
# =============================================================================

# This WILL WORK (centralized config)
python scripts/train.py train --experiment baseline

# This will show you available experiments if you typo
python scripts/train.py train --experiment typo
# Error: Invalid value for '--experiment': invalid choice: typo. 
# (choose from baseline, aggressive, debug, production)

# This will work - override any nested config
python scripts/train.py train \
    --experiment baseline \
    --override training.optimizer.weight_decay=0.02 \
    --override text_processing.max_input_length=256

=============================================================================
# 2. OVERRIDING SPECIFIC PARAMETERS
# =============================================================================

# Override max_epochs using the new dedicated flag
python scripts/train.py train --experiment baseline --max-epochs 12

# Override batch_size using the new dedicated flag
python scripts/train.py train --experiment baseline --batch-size 32

# Override learning_rate using the --override syntax
python scripts/train.py train --experiment baseline --override training.optimizer.lr=1e-5

# Combine multiple overrides
python scripts/train.py train \
    --experiment baseline \
    --max-epochs 10 \
    --batch-size 24 \
    --override training.optimizer.lr=8e-6
Update Section 6: Resume Training
The example for resuming and changing settings should use the correct override syntax.

Replace the relevant part of Section 6 with this:

Markdown

# Resume but change some settings
python scripts/train.py train \
    --experiment baseline \
    --resume-from outputs/models/last.ckpt \
    --override training.optimizer.lr=3e-6 \
    --override training.early_stopping.patience=1
Update Section 8: Comparison Experiments
This section also uses the old, incorrect --learning-rate flag.

Replace the "Compare different learning rates" example in Section 8 with this:

Markdown

# Compare different learning rates
python scripts/train.py train \
    --experiment baseline \
    --override training.optimizer.lr=3e-6 \
    --override wandb.name=baseline_lr3e6

python scripts/train.py train \
    --experiment baseline \
    --override training.optimizer.lr=1e-5 \
    --override wandb.name=baseline_lr1e5
Update Section 10: Migration from Old System
This section is now even more relevant. It should clearly show that fragmented configs and old flags are deprecated in favor of the new centralized experiment system.

Replace Section 10 with this:

Markdown

# =============================================================================
# 10. MIGRATION FROM OLD SYSTEM
# =============================================================================

# OLD WAY (individual flags, now removed):
# python scripts/train.py train --config-name kobart-base-v2 --learning-rate 5e-5

# NEW WAY (centralized experiment with overrides):
python scripts/train.py train \
    --experiment baseline \
    --override training.optimizer.lr=5e-5

# OLD WAY (unclear which config is being run):
# python scripts/train.py --config-name training/aggressive

# NEW WAY (clear experiment definition):
python scripts/train.py train --experiment aggressive

# =============================================================================
# 11. DATA ANALYSIS SCRIPT
# =============================================================================

You can now run these functions directly from your terminal.

**To see statistics about your validation set:**

```bash
python scripts/analyze_data.py analyze --split val
```

**To print the 5th sample from the training set:**

```bash
python scripts/analyze_data.py sample --split train --index 4
```