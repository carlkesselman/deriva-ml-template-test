"""Define experiments.

Experiments are pre-configured combinations of model, dataset, and asset settings.
They use Hydra's defaults list to override specific config groups and inherit from
the main DerivaModelConfig.

Usage:
    # Run a single experiment
    uv run deriva-ml-run +experiment=vgg19_quick

    # Run multiple experiments using a multirun config
    uv run deriva-ml-run +multirun=quick_vs_extended

    # Override experiment settings
    uv run deriva-ml-run +experiment=vgg19_quick datasets=graded_training

For hyperparameter sweeps and grid searches, use multirun configs defined in
configs/multiruns.py - they are self-contained and don't require separate
experiment definitions.

Reference:
    https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/configuring_experiments.html
"""

from hydra_zen import make_config, store

from configs.base import DerivaModelConfig

# Use _global_ package to allow overrides at the root level
experiment_store = store(group="experiment", package="_global_")

# =============================================================================
# VGG19 Glaucoma Classification Experiments
# =============================================================================
# These experiments use the VGG19 model for glaucoma classification.
# Each experiment inherits from DerivaModelConfig (a builds() of run_model)
# and overrides specific config groups.

# -----------------------------------------------------------------------------
# Quick experiments (for testing and debugging)
# -----------------------------------------------------------------------------

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "vgg19_quick"},
            {"override /datasets": "test_small"},
        ],
        description="Quick VGG19 training: 3 epochs on small test dataset for fast validation",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_quick",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "default_model"},
            {"override /datasets": "test_small"},
        ],
        description="Default VGG19 training: 10 epochs on small test dataset",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_default",
)

# -----------------------------------------------------------------------------
# 10% subset experiments (for quick iteration)
# -----------------------------------------------------------------------------

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "vgg19_quick"},
            {"override /datasets": "subset_10pct_combined"},
        ],
        description="Quick VGG19 on 10% subset: 3 epochs for rapid prototyping",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_quick_10pct",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "default_model"},
            {"override /datasets": "subset_10pct_combined"},
        ],
        description="Default VGG19 on 10% subset: 10 epochs for testing",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_default_10pct",
)

# -----------------------------------------------------------------------------
# Graded dataset experiments (balanced classes)
# -----------------------------------------------------------------------------

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "vgg19_quick"},
            {"override /datasets": "graded_combined"},
        ],
        description="Quick VGG19 on graded datasets: 3 epochs, balanced classes",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_quick_graded",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "default_model"},
            {"override /datasets": "graded_combined"},
        ],
        description="Default VGG19 on graded datasets: 10 epochs, balanced classes",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_default_graded",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "vgg19_extended"},
            {"override /datasets": "graded_combined"},
        ],
        description="Extended VGG19 on graded datasets: 50 epochs with regularization",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_extended_graded",
)

# -----------------------------------------------------------------------------
# Frozen feature experiments (transfer learning)
# -----------------------------------------------------------------------------

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "vgg19_frozen"},
            {"override /datasets": "graded_combined"},
        ],
        description="Frozen VGG19 features: Only train classifier on graded data",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_frozen_graded",
)

# -----------------------------------------------------------------------------
# Full dataset experiments (production training)
# -----------------------------------------------------------------------------

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "vgg19_quick"},
            {"override /datasets": "full_combined"},
        ],
        description="Quick VGG19 on full dataset: 3 epochs for baseline validation",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_quick_full",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "vgg19_extended"},
            {"override /datasets": "full_combined"},
        ],
        description="Extended VGG19 on full dataset: 50 epochs, production training",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_extended_full",
)

# -----------------------------------------------------------------------------
# Test-only experiments (evaluation)
# -----------------------------------------------------------------------------

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "vgg19_test_only"},
            {"override /datasets": "graded_test"},
            # Note: assets must be specified at runtime with pre-trained weights
        ],
        description="VGG19 evaluation only: load pre-trained weights and evaluate on graded test set",
        bases=(DerivaModelConfig,),
    ),
    name="vgg19_test_only",
)
