"""Dataset Configuration.

This module defines dataset configurations for model training and evaluation.

Configuration Group: datasets
-----------------------------
Datasets are specified as lists of DatasetSpecConfig objects, where each object
identifies a dataset by its RID and optionally a version. Multiple datasets can
be combined into a single configuration for training on multiple data sources.

REQUIRED: A configuration named "default_dataset" must be defined.
This is used as the default dataset when no override is specified.

Example usage:
    # Use default dataset
    uv run deriva-ml-run

    # Use a specific dataset
    uv run deriva-ml-run datasets=graded_training

    # Combine multiple datasets
    datasets_combined = [
        DatasetSpecConfig(rid="ABC1", version="1.0.0"),
        DatasetSpecConfig(rid="ABC2", version="1.0.0"),
    ]
"""
from hydra_zen import store
from deriva_ml.dataset import DatasetSpecConfig
from deriva_ml.execution import with_description

# ---------------------------------------------------------------------------
# Dataset Configurations
# ---------------------------------------------------------------------------
# Configure a list of datasets by specifying the RID and version of each
# dataset that goes into the collection. The group name "datasets" must
# match the parameter name in BaseConfig.

# =============================================================================
# Eye-AI Glaucoma Classification Datasets (dev.eye-ai.org, schema: eye-ai)
# =============================================================================

datasets_store = store(group="datasets")

# Special config for notebooks that don't need datasets (e.g., ROC analysis)
datasets_store(
    [],
    name="no_datasets",
)

# -----------------------------------------------------------------------------
# Small test dataset (for quick testing and debugging)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="5-XW4J", version="0.3.0")],
        "Small image test dataset for quick debugging and validation. "
        "Use for rapid prototyping and testing model code.",
    ),
    name="test_small",
)

# -----------------------------------------------------------------------------
# 10% Subset datasets (for quick experiments)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2-A5T0", version="3.3.0")],
        "10% Training Dataset - subset of the full training dataset. "
        "Use for quick experiments and hyperparameter tuning.",
    ),
    name="subset_10pct_training",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2-A5T4", version="3.3.0")],
        "10% Test Dataset - subset of the full test dataset. "
        "Use for quick evaluation during development.",
    ),
    name="subset_10pct_test",
)

datasets_store(
    with_description(
        [
            DatasetSpecConfig(rid="2-A5T0", version="3.3.0"),
            DatasetSpecConfig(rid="2-A5T4", version="3.3.0"),
        ],
        "10% Train + Test combined - both subsets for quick training and evaluation.",
    ),
    name="subset_10pct_combined",
)

# -----------------------------------------------------------------------------
# Graded datasets (balanced by diagnosis category)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2-36BW", version="9.3.0")],
        "Graded Training Dataset - 800 subjects per diagnosis category. "
        "Balanced subset from the full training dataset. "
        "Use for training with balanced class distribution.",
    ),
    name="graded_training",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2-277M", version="5.3.0")],
        "Graded Test Dataset - 250 subjects per diagnosis category. "
        "Balanced subset for model evaluation.",
    ),
    name="graded_test",
)

datasets_store(
    with_description(
        [
            DatasetSpecConfig(rid="2-36BW", version="9.3.0"),
            DatasetSpecConfig(rid="2-277M", version="5.3.0"),
        ],
        "Graded Train + Test combined - balanced datasets for full training pipeline. "
        "Training: 800 subjects/category, Test: 250 subjects/category.",
    ),
    name="graded_combined",
)

# -----------------------------------------------------------------------------
# Full LAC datasets (unbalanced, complete data)
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2-277G", version="5.3.0")],
        "Full Training Dataset - 75% of development dataset. "
        "Unbalanced class distribution. Use for production training.",
    ),
    name="full_training",
)

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2-277C", version="3.3.0")],
        "Full Test Dataset - 20% of LAC subjects per diagnosis category. "
        "Use for production model evaluation.",
    ),
    name="full_test",
)

datasets_store(
    with_description(
        [
            DatasetSpecConfig(rid="2-277G", version="5.3.0"),
            DatasetSpecConfig(rid="2-277C", version="3.3.0"),
        ],
        "Full Train + Test combined - complete LAC datasets for production training.",
    ),
    name="full_combined",
)

# -----------------------------------------------------------------------------
# LAC Complete dataset
# -----------------------------------------------------------------------------

datasets_store(
    with_description(
        [DatasetSpecConfig(rid="2-1S12", version="5.3.0")],
        "Complete LAC Dataset - all 7021 subjects. "
        "Use for data exploration or when you need all available data.",
    ),
    name="lac_complete",
)

# -----------------------------------------------------------------------------
# REQUIRED: default_dataset - used when no dataset is specified
# -----------------------------------------------------------------------------
# Using small test dataset as default for quick iteration during development

datasets_store(
    [DatasetSpecConfig(rid="5-XW4J", version="0.3.0")],
    name="default_dataset",
)
