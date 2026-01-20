"""Configuration for ROC analysis notebook.

This module defines the hydra-zen configuration for the ROC curve analysis notebook.

Usage:
    In notebook:

        from deriva_ml.execution import run_notebook

        ml, execution, config = run_notebook("roc_analysis")
        # Ready to use!
        # - config.assets: probability file RIDs
        # - config.show_per_class: whether to show individual class curves

    From command line:

        # Run with specific configuration
        deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
            --host dev.eye-ai.org --catalog eye-ai \\
            --config roc_analysis

        # Or override assets directly
        deriva-ml-run-notebook notebooks/roc_analysis.ipynb \\
            assets=<your_asset_config>

Available Configurations:
    - roc_analysis: Default ROC curve analysis

Note: Asset RIDs for prediction probability files should be added to
configs/assets.py after running VGG19 experiments.

Configuration Groups:
    - deriva_ml: DerivaML connection settings (default_deriva, eye_ai, etc.)
    - assets: Asset RID lists for probability files
"""

from dataclasses import dataclass

from deriva_ml.execution import BaseConfig, notebook_config


@dataclass
class ROCAnalysisConfig(BaseConfig):
    """Configuration for ROC analysis notebook.

    Attributes:
        show_per_class: If True, plot individual ROC curves for each class.
            If False, only show micro/macro averaged curves.
        confidence_threshold: Minimum confidence threshold for predictions
            to be included in the analysis (0.0 to 1.0).
    """

    show_per_class: bool = True
    confidence_threshold: float = 0.0


# =============================================================================
# ROC Analysis Notebook Configurations
# =============================================================================

# Default: Use no assets (must be specified at runtime after experiments are run)
notebook_config(
    "roc_analysis",
    config_class=ROCAnalysisConfig,
    defaults={"assets": "no_assets", "datasets": "no_datasets"},
    description="ROC curve analysis for VGG19 glaucoma classification",
)

# -----------------------------------------------------------------------------
# Additional Configurations (add after running experiments)
# -----------------------------------------------------------------------------
# After running VGG19 experiments and obtaining prediction probability files,
# add their RIDs to configs/assets.py and create configurations here.
#
# Example:
#
# notebook_config(
#     "roc_vgg19_comparison",
#     config_class=ROCAnalysisConfig,
#     defaults={"assets": "roc_vgg19_comparison", "datasets": "no_datasets"},
#     description="ROC analysis: VGG19 model comparison",
# )
