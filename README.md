# VGG19 Glaucoma Classification

A VGG19-based model for binary glaucoma classification on fundus images, integrated with DerivaML for reproducible ML workflows.

## Overview

This project uses transfer learning with VGG19 (pretrained on ImageNet) to classify fundus images as:
- **No Glaucoma**: Healthy eye
- **Suspected Glaucoma**: Glaucoma indicators present

Images labeled "Unknown" are automatically filtered out for clean binary classification.

## Quick Start

### 1. Initialize Environment

```bash
# Create environment and install dependencies
uv sync

# For notebook support
uv sync --group=jupyter
uv run nbstripout --install
uv run deriva-ml-install-kernel
```

### 2. Authenticate

```bash
uv run deriva-globus-auth-utils login --host dev.eye-ai.org
```

### 3. Run

```bash
# Quick test run (3 epochs, small dataset)
uv run deriva-ml-run +experiment=vgg19_quick

# Dry run (no catalog writes)
uv run deriva-ml-run +experiment=vgg19_quick dry_run=true

# Full training on graded (balanced) dataset
uv run deriva-ml-run +experiment=vgg19_extended_graded

# Show available configurations
uv run deriva-ml-run --info
```

## Project Layout

```
.
├─ pyproject.toml                  # Project metadata and dependencies
├─ src/
│  ├─ models/
│  │  └─ vgg19_glaucoma.py        # VGG19 model implementation
│  └─ configs/                     # Hydra-zen configurations
│     ├─ deriva.py                 # Connection settings (dev.eye-ai.org)
│     ├─ datasets.py               # Dataset specifications
│     ├─ vgg19_glaucoma.py        # Model config variants
│     ├─ experiments.py            # Experiment presets
│     ├─ multiruns.py             # Hyperparameter sweeps
│     └─ assets.py                 # Asset configurations
├─ notebooks/
│  └─ roc_analysis.ipynb          # ROC curve analysis
└─ CLAUDE.md                       # AI assistant guidance
```

## Available Datasets

| Dataset | Description | Use Case |
|---------|-------------|----------|
| `test_small` | Small test dataset | Quick debugging |
| `subset_10pct_combined` | 10% of training + test | Rapid prototyping |
| `graded_combined` | Balanced (800 train / 250 test per class) | Recommended for training |
| `full_combined` | Complete LAC dataset | Production training |

## Model Configurations

| Config | Description |
|--------|-------------|
| `default_model` | Pretrained VGG19, 10 epochs, lr=1e-4 |
| `vgg19_quick` | Fast training (3 epochs) |
| `vgg19_frozen` | Freeze features, train classifier only |
| `vgg19_regularized` | Heavy regularization (dropout 0.7) |
| `vgg19_extended` | Extended training (50 epochs) |
| `vgg19_test_only` | Evaluation only with pretrained weights |

## Experiments

```bash
# Quick experiments
uv run deriva-ml-run +experiment=vgg19_quick           # Small test dataset
uv run deriva-ml-run +experiment=vgg19_quick_10pct     # 10% subset
uv run deriva-ml-run +experiment=vgg19_quick_graded    # Balanced dataset

# Production training
uv run deriva-ml-run +experiment=vgg19_extended_graded # Extended on balanced
uv run deriva-ml-run +experiment=vgg19_extended_full   # Extended on full data

# Transfer learning comparison
uv run deriva-ml-run +experiment=vgg19_frozen_graded   # Frozen features
```

## Hyperparameter Sweeps

```bash
# Compare training configurations
uv run deriva-ml-run +multirun=quick_vs_extended

# Learning rate sweep
uv run deriva-ml-run +multirun=lr_sweep

# Epoch sweep
uv run deriva-ml-run +multirun=epoch_sweep

# Frozen vs fine-tuning comparison
uv run deriva-ml-run +multirun=frozen_vs_finetune
```

## Overriding Parameters

```bash
# Override model parameters
uv run deriva-ml-run model_config.epochs=20 model_config.learning_rate=0.0001

# Override dataset
uv run deriva-ml-run +experiment=vgg19_quick datasets=graded_combined

# Override connection
uv run deriva-ml-run --host www.eye-ai.org --catalog eye-ai +experiment=vgg19_quick
```

## Versioning

Create version tags before significant runs:

```bash
uv run bump-version patch   # Bug fixes
uv run bump-version minor   # New features
uv run bump-version major   # Breaking changes
```

## Further Reading

- [DerivaML Library](https://informatics-isi-edu.github.io/deriva-ml/) - Core library documentation
- [Hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) - Configuration framework
- [Eye-AI Catalog](https://dev.eye-ai.org) - Data source
