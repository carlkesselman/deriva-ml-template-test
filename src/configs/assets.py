"""Asset Configuration.

This module defines asset configurations for execution (model weights, checkpoints, etc.).

Configuration Group: assets
---------------------------
Assets are additional files needed beyond the dataset. They are specified as
lists of Resource IDs (RIDs) and automatically downloaded when the execution
is initialized.

Typical assets include:
- Pre-trained model weights
- Model checkpoints
- Configuration files
- Reference data files

REQUIRED: A configuration named "default_asset" must be defined.
This is used as the default (typically an empty list) when no override is specified.

Example usage:
    # Use default (no assets)
    uv run deriva-ml-run

    # Use specific assets
    uv run deriva-ml-run assets=vgg19_trained_weights

Configuration Format:
    asset_store(
        with_description(
            ["RID1", "RID2"],
            "Description of what these assets are for",
        ),
        name="my_asset_config",
    )
"""

from hydra_zen import store
from deriva_ml.execution import with_description

# ---------------------------------------------------------------------------
# Asset Store
# ---------------------------------------------------------------------------
asset_store = store(group="assets")

# REQUIRED: default_asset - used when no assets are specified (typically empty)
# Note: Using plain list (not with_description) because this is used as a merge base
# in notebook configs. with_description creates a DictConfig which can't merge with
# the ListConfig default in BaseConfig.
asset_store(
    [],
    name="default_asset",
)

# Alias for clarity in notebook configs
asset_store(
    [],
    name="no_assets",
)

# =============================================================================
# Eye-AI VGG19 Glaucoma Model Assets (dev.eye-ai.org, schema: eye-ai)
# =============================================================================
# Note: Asset RIDs will be populated after initial model training runs.
# Once you have trained models, add their weight files here.
#
# Example format:
#
# asset_store(
#     with_description(
#         ["<RID>"],
#         "VGG19 weights trained on graded dataset for 50 epochs. "
#         "Accuracy: XX%. Source: execution <EXEC_RID>.",
#     ),
#     name="vgg19_graded_weights",
# )
#
# asset_store(
#     with_description(
#         ["<RID1>", "<RID2>"],
#         "Prediction probability files from VGG19 experiments for ROC analysis.",
#     ),
#     name="roc_vgg19_comparison",
# )

# -----------------------------------------------------------------------------
# Placeholder for future trained model weights
# -----------------------------------------------------------------------------
# After running experiments, add asset configurations here.
# Use `uv run deriva-ml-run +experiment=vgg19_extended_graded` to train,
# then find the model weight RIDs in the execution outputs.
