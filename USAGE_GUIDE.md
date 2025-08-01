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

# Override learning rate for baseline experiment
python scripts/train.py train --experiment baseline --learning-rate 1e-5

# Override batch size
python scripts/train.py train --experiment baseline --batch-size 32

# Override max epochs
python scripts/train.py train --experiment baseline --max-epochs 12

# Multiple simple overrides
python scripts/train.py train \
    --experiment baseline \
    --learning-rate 8e-6 \
    --batch-size 24 \
    --max-epochs 10

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
    --learning-rate 3e-6 \
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

# =============================================================================
# 10. MIGRATION FROM OLD SYSTEM
# =============================================================================

# OLD WAY (fragmented configs):
# python scripts/train.py --config-name kobart-base-v2 --override training.generation.max_length=80

# NEW WAY (centralized):
python scripts/train.py train \
    --experiment baseline \
    --override generation.max_length=80

# OLD WAY (unclear which config):
# python scripts/train.py --config-name training/aggressive

# NEW WAY (clear experiment):
python scripts/train.py train --experiment aggressive