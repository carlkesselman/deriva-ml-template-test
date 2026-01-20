"""Model Configuration (VGG19 Glaucoma Classification).

This module defines model configurations for the VGG19 glaucoma classifier.

Configuration Group: model_config
---------------------------------
Model configurations define the hyperparameters and settings for your ML model.
Each configuration is a set of parameters that can be selected at runtime.

REQUIRED: A configuration named "default_model" must be defined.
This is used as the default model configuration when no override is specified.

All model parameters are configurable via Hydra:
- Architecture: pretrained, freeze_features, dropout_rate
- Training: learning_rate, epochs, batch_size, weight_decay
- Image: image_size

Example usage:
    # Run with default config
    uv run deriva-ml-run

    # Run with a specific model config
    uv run deriva-ml-run model_config=vgg19_frozen

    # Override specific parameters
    uv run deriva-ml-run model_config.epochs=20 model_config.learning_rate=0.0001
"""
from __future__ import annotations

from hydra_zen import builds, store

from models.vgg19_glaucoma import vgg19_glaucoma

# Build the base VGG19 Glaucoma configuration.
# All parameters have sensible defaults for transfer learning.
VGG19GlaucomaConfig = builds(
    vgg19_glaucoma,
    # Architecture parameters
    pretrained=True,
    freeze_features=False,
    dropout_rate=0.5,
    # Training parameters
    learning_rate=1e-4,
    epochs=10,
    batch_size=32,
    weight_decay=1e-4,
    # Image parameters
    image_size=224,
    # Hydra-zen settings
    populate_full_signature=True,
    zen_partial=True,  # Execution context added later
)

# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------
# The group name "model_config" must match the parameter name in BaseConfig.

model_store = store(group="model_config")

# REQUIRED: default_model - used when no model config is specified
model_store(
    VGG19GlaucomaConfig,
    name="default_model",
    zen_meta={
        "description": (
            "Default VGG19 glaucoma classifier: pretrained ImageNet weights, "
            "fine-tune all layers, 10 epochs, lr=1e-4. "
            "Balanced config for transfer learning on fundus images."
        )
    },
)

# Quick training - fewer epochs for testing
model_store(
    VGG19GlaucomaConfig,
    name="vgg19_quick",
    epochs=3,
    batch_size=16,
    zen_meta={
        "description": (
            "Quick training config: 3 epochs, batch 16. Use for rapid iteration, "
            "debugging, and verifying the training pipeline works correctly."
        )
    },
)

# Frozen features - only train classifier
model_store(
    VGG19GlaucomaConfig,
    name="vgg19_frozen",
    freeze_features=True,
    learning_rate=1e-3,
    epochs=15,
    zen_meta={
        "description": (
            "Frozen feature extractor: Only train the classifier head. "
            "Faster training, less GPU memory, good for small datasets. "
            "Higher learning rate since fewer parameters to update."
        )
    },
)

# Full fine-tuning with regularization
model_store(
    VGG19GlaucomaConfig,
    name="vgg19_regularized",
    dropout_rate=0.7,
    weight_decay=1e-3,
    epochs=20,
    zen_meta={
        "description": (
            "Regularized config: 70% dropout, weight decay 1e-3, 20 epochs. "
            "Use to reduce overfitting when training on smaller datasets."
        )
    },
)

# Fast learning rate - aggressive fine-tuning
model_store(
    VGG19GlaucomaConfig,
    name="vgg19_fast_lr",
    learning_rate=1e-3,
    epochs=5,
    freeze_features=True,
    zen_meta={
        "description": (
            "Fast learning rate (1e-3) with frozen features: "
            "Quickly train classifier, then unfreeze for fine-tuning. "
            "Good as first stage in two-phase training."
        )
    },
)

# Slow learning rate - careful fine-tuning
model_store(
    VGG19GlaucomaConfig,
    name="vgg19_slow_lr",
    learning_rate=1e-5,
    epochs=30,
    zen_meta={
        "description": (
            "Slow learning rate (1e-5): More stable convergence, 30 epochs. "
            "Use when training is unstable or for careful fine-tuning."
        )
    },
)

# Extended training - for best accuracy
model_store(
    VGG19GlaucomaConfig,
    name="vgg19_extended",
    dropout_rate=0.6,
    weight_decay=1e-4,
    learning_rate=1e-4,
    epochs=50,
    zen_meta={
        "description": (
            "Extended training for best accuracy: Full fine-tuning with "
            "regularization (dropout 0.6, weight decay 1e-4), 50 epochs. "
            "Use for final production training when accuracy is priority."
        )
    },
)

# Large batch - for systems with more GPU memory
model_store(
    VGG19GlaucomaConfig,
    name="vgg19_large_batch",
    batch_size=64,
    learning_rate=2e-4,
    epochs=15,
    zen_meta={
        "description": (
            "Large batch training: batch size 64, scaled learning rate. "
            "Requires more GPU memory but can be faster per epoch. "
            "Use on systems with 16GB+ GPU memory."
        )
    },
)

# Small batch - for memory-constrained systems
model_store(
    VGG19GlaucomaConfig,
    name="vgg19_small_batch",
    batch_size=8,
    learning_rate=5e-5,
    epochs=20,
    zen_meta={
        "description": (
            "Small batch training: batch size 8, reduced learning rate. "
            "For memory-constrained systems or very large images. "
            "More epochs to compensate for fewer updates per epoch."
        )
    },
)

# Test-only mode - load weights and run evaluation without training
model_store(
    VGG19GlaucomaConfig,
    name="vgg19_test_only",
    test_only=True,
    weights_filename="vgg19_glaucoma_weights.pt",
    zen_meta={
        "description": (
            "Test-only mode: Skips training, loads pretrained weights from assets, "
            "runs inference on dataset. Requires assets with vgg19_glaucoma_weights.pt. "
            "Use for evaluation and generating predictions on new data."
        )
    },
)
