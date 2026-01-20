"""Workflow Configuration.

This module defines workflow configurations for DerivaML executions.

Configuration Group: workflow
-----------------------------
A Workflow represents a computational pipeline and its metadata. It describes
what the code does (name, description, type) and automatically captures Git
repository information (URL, checksum, version) for provenance tracking.

REQUIRED: A configuration named "default_workflow" must be defined.
This is used as the default workflow when no override is specified.

Example usage:
    # Use default workflow
    uv run deriva-ml-run

    # Use a specific workflow
    uv run deriva-ml-run workflow=vgg19_glaucoma
"""

from hydra_zen import store, builds

from deriva_ml.execution import Workflow

# ---------------------------------------------------------------------------
# Workflow Configurations
# ---------------------------------------------------------------------------

# VGG19 Glaucoma Classification workflow - default for this project
VGG19GlaucomaWorkflow = builds(
    Workflow,
    name="VGG19 Glaucoma Classification",
    workflow_type="VGG19 Catalog Model training",
    description="""
Train a VGG19 model for binary glaucoma classification on fundus images.

## Task
Binary classification of fundus images:
- **No Glaucoma**: Healthy eye
- **Suspected Glaucoma**: Glaucoma indicators present

## Architecture
- **Backbone**: VGG19 pretrained on ImageNet
- **Classifier**: 4096 → 4096 → 2 (binary output)
- **Input**: 224×224 RGB fundus images

## Features
- Transfer learning with ImageNet pretrained weights
- Configurable: freeze features, dropout, learning rate schedules
- Automatic data loading from DerivaML datasets via `restructure_assets()`
- Filters out "Unknown" labels for clean binary classification
- Outputs: model weights (`.pt`) and training log

## Data Requirements
Datasets must contain Image assets with Image_Diagnosis feature values.
Images labeled "Unknown" are automatically filtered out.

## Expected Performance
Depends on dataset size and quality. Typical range: 80-95% accuracy.
""".strip(),
    populate_full_signature=True,
)


# ---------------------------------------------------------------------------
# Register with Hydra-Zen Store
# ---------------------------------------------------------------------------

workflow_store = store(group="workflow")

# REQUIRED: default_workflow - used when no workflow is specified
workflow_store(VGG19GlaucomaWorkflow, name="default_workflow")

# Additional workflow configurations
workflow_store(VGG19GlaucomaWorkflow, name="vgg19_glaucoma")
