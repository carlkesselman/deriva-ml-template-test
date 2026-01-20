"""Sweep configurations for multi-experiment runs.

Sweeps provide rich markdown descriptions for the parent execution when running
multiple experiments together. The sweep config documents why experiments are
being compared and what outcomes are expected.

Unlike ad-hoc parameter sweeps (e.g., `--multirun model_config.epochs=10,20,50`),
sweep configs let you:
- Document why experiments are being compared
- Use full markdown formatting in descriptions (tables, headers, bold, etc.)
- Track sweep metadata in the parent execution
- Keep the rationale alongside the code

Usage:
    # Run a pre-defined sweep (specify experiments on command line)
    uv run deriva-ml-run --multirun \\
        +experiment=vgg19_quick,vgg19_extended \\
        sweep_description="$(<src/configs/sweep_descriptions/quick_vs_extended.md)"

    # Or set sweep_description in your experiment config directly
"""

# =============================================================================
# Sweep Descriptions
# =============================================================================
# These are standalone markdown descriptions that can be used with any multirun.
# They are defined here as Python strings for easy reference and documentation.

QUICK_VS_EXTENDED_DESCRIPTION = """## VGG19 Glaucoma Multi-Experiment Comparison

**Objective:** Compare model performance across two training configurations to evaluate
the trade-off between training speed and model accuracy for glaucoma classification.

### Experiments

| Experiment | Epochs | Configuration | Regularization | Dataset |
|------------|--------|---------------|----------------|---------|
| `vgg19_quick` | 3 | Pretrained, fine-tune all | Dropout 0.5 | Test Small |
| `vgg19_extended` | 50 | Pretrained, fine-tune all | Dropout 0.6, Weight Decay 1e-4 | Test Small |

### Configuration Details

**vgg19_quick** - Fast validation baseline
- VGG19 pretrained on ImageNet
- All layers trainable
- Learning rate: 1e-4
- Batch size: 16

**vgg19_extended** - Production-quality training
- VGG19 pretrained on ImageNet
- All layers trainable
- Learning rate: 1e-4
- Dropout: 0.6
- Weight decay: 1e-4

### Expected Outcomes

- The quick model should train in a few minutes but may have lower accuracy
- The extended model should achieve higher accuracy with more stable convergence
- This comparison helps validate the training pipeline before running on full data
"""

GRADED_DATASET_DESCRIPTION = """## VGG19 Graded Dataset Comparison

**Objective:** Evaluate VGG19 model on the balanced graded datasets to get
realistic performance estimates for glaucoma classification.

### Experiments

| Experiment | Epochs | Configuration | Dataset |
|------------|--------|---------------|---------|
| `vgg19_quick_graded` | 3 | Quick training | Graded (balanced) |
| `vgg19_extended_graded` | 50 | Extended training | Graded (balanced) |

### Notes

- Graded datasets are balanced by diagnosis category (800 train / 250 test per class)
- Extended model should show good generalization with balanced data
- Use this sweep for validation before production training
"""

FULL_DATASET_DESCRIPTION = """## VGG19 Full Dataset Comparison

**Objective:** Evaluate VGG19 model on the complete LAC dataset
to get production-level performance estimates.

### Experiments

| Experiment | Epochs | Configuration | Dataset |
|------------|--------|---------------|---------|
| `vgg19_quick_full` | 3 | Quick training | Full LAC |
| `vgg19_extended_full` | 50 | Extended training | Full LAC |

### Notes

- Full dataset runs take significantly longer
- Extended model should achieve best accuracy with more training data
- Use this sweep for final model selection before production deployment
"""

# =============================================================================
# Learning Rate Sweep
# =============================================================================
# Explores the effect of learning rate on model convergence and final accuracy

LEARNING_RATE_SWEEP_DESCRIPTION = """## Learning Rate Hyperparameter Sweep

**Objective:** Identify the optimal learning rate for VGG19 fine-tuning on fundus images
by comparing training dynamics across a range of values.

### Experiments

| Learning Rate | Expected Behavior |
|--------------|-------------------|
| 0.00001 | Very slow convergence, may underfit in limited epochs |
| 0.0001 | Standard rate for fine-tuning, good balance |
| 0.001 | Fast convergence, may overshoot for fine-tuning |
| 0.01 | Aggressive, likely unstable for pretrained weights |

### Methodology

All experiments use the same:
- Architecture: VGG19 pretrained, fine-tune all layers
- Dataset: Graded combined (balanced classes)
- Epochs: 10 (enough to observe convergence behavior)
- Batch size: 32

### Metrics to Compare

- Training loss curve shape
- Final training/test accuracy
- Presence of training instability (loss spikes)
- Generalization gap (train vs test accuracy)

### Command

```bash
uv run deriva-ml-run --multirun \\
    +experiment=vgg19_default_graded \\
    model_config.learning_rate=0.00001,0.0001,0.001,0.01
```
"""

# =============================================================================
# Epoch Sweep
# =============================================================================
# Explores the effect of training duration on model performance

EPOCH_SWEEP_DESCRIPTION = """## Training Duration (Epochs) Sweep

**Objective:** Analyze how training duration affects VGG19 performance and identify
the point of diminishing returns or overfitting onset.

### Experiments

| Epochs | Expected Behavior |
|--------|-------------------|
| 5 | Underfitting, model still adapting pretrained features |
| 10 | Reasonable performance, may still be improving |
| 25 | Good performance, watch for overfitting signs |
| 50 | Extended training, maximum accuracy potential |

### Methodology

All experiments use the same:
- Architecture: VGG19 pretrained, fine-tune all layers
- Dataset: Graded combined (balanced classes)
- Learning rate: 1e-4
- Batch size: 32
- Regularization: Dropout 0.5, weight decay 1e-4

### Metrics to Compare

- Training vs test accuracy divergence (overfitting indicator)
- Final test accuracy plateau
- Training time per epoch
- Optimal early stopping point

### Command

```bash
uv run deriva-ml-run --multirun \\
    +experiment=vgg19_default_graded \\
    model_config.epochs=5,10,25,50
```
"""

# =============================================================================
# Frozen vs Fine-Tuning Comparison
# =============================================================================
# Compares transfer learning strategies

FROZEN_VS_FINETUNE_DESCRIPTION = """## Transfer Learning Strategy Comparison

**Objective:** Compare frozen feature extraction vs full fine-tuning to determine
the best transfer learning approach for glaucoma classification.

### Experiments

| Strategy | Configuration | Expected Behavior |
|----------|---------------|-------------------|
| Frozen | Only train classifier | Fast training, may underfit |
| Fine-tune | Train all layers | Slower training, better adaptation |

### Methodology

**Frozen Features:**
- VGG19 convolutional layers frozen
- Only classifier head is trained
- Higher learning rate possible (1e-3)
- Less GPU memory required

**Full Fine-Tuning:**
- All layers trainable
- Lower learning rate (1e-4) to preserve pretrained features
- More GPU memory required
- Better domain adaptation

### When to Use Each

- **Frozen:** Small datasets, limited compute, quick iteration
- **Fine-tune:** Larger datasets, domain shift from ImageNet, best accuracy needed

### Command

```bash
uv run deriva-ml-run --multirun \\
    +experiment=vgg19_frozen_graded,vgg19_default_graded
```
"""
