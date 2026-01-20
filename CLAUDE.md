# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGG19 Glaucoma Classification - a DerivaML-integrated model for binary glaucoma classification on fundus images. Uses VGG19 pretrained on ImageNet with fine-tuning for glaucoma detection. The model classifies images as either "No Glaucoma" or "Suspected Glaucoma", filtering out "Unknown" labels.

## Common Commands

```bash
# Environment setup
uv sync                                    # Initialize environment
uv sync --group=jupyter                   # Add Jupyter support
uv sync --group=pytorch                   # Add PyTorch support

# Running models (uses Hydra config defaults for host/catalog)
uv run deriva-ml-run                                  # Run with defaults
uv run deriva-ml-run model_config=vgg19_frozen       # Override model config
uv run deriva-ml-run +experiment=vgg19_quick         # Use experiment preset
uv run deriva-ml-run dry_run=true                    # Dry run (no catalog writes)
uv run deriva-ml-run --multirun +experiment=vgg19_quick_graded,vgg19_extended_graded  # Multiple experiments
uv run deriva-ml-run --info                          # Show available configs

# Override host/catalog from command line
uv run deriva-ml-run --host dev.eye-ai.org --catalog eye-ai +experiment=vgg19_quick

# Notebook execution (uses Hydra config defaults for host/catalog)
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb assets=my_assets
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb --info

# Override host/catalog from command line
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \
  --host dev.eye-ai.org --catalog eye-ai

# Linting and formatting
uv run ruff check src/
uv run ruff format src/

# Testing
uv run pytest

# Version management
uv run bump-version major|minor|patch
uv run python -m setuptools_scm

# Authentication
uv run deriva-globus-auth-utils login --host dev.eye-ai.org
```

## Architecture

### Configuration System (Hydra-Zen)

All configuration is Python-first using hydra-zen, no YAML files. Configs are in `src/configs/`:
- `deriva.py` - DerivaML connection configs (dev.eye-ai.org, www.eye-ai.org)
- `datasets.py` - Dataset specifications (test_small, graded, full, etc.)
- `assets.py` - Asset RID configurations (model weights, predictions)
- `workflow.py` - Workflow definitions
- `vgg19_glaucoma.py` - Model variant configs (9 variants)
- `experiments.py` - Experiment presets

### Model Pattern

Models follow this signature pattern:
```python
def model_name(
    param1: type = default,
    param2: type = default,
    # Always present - injected by framework
    ml_instance: DerivaML = None,
    execution: Execution | None = None,
) -> None:
```

Wrap with `builds(..., zen_partial=True)` for deferred execution:
```python
from hydra_zen import builds, store

ModelConfig = builds(model_function, param1=val1, zen_partial=True)
store(ModelConfig, group="model_config", name="default")
store(ModelConfig, param1=val2, group="model_config", name="variant")
```

### Entry Point

`deriva-ml-run` - CLI provided by deriva-ml. Loads config modules from `src/configs/` automatically.

### Notebook Configuration Pattern

Notebooks use the simplified `run_notebook()` API for initialization:

1. **Define a config module** in `src/configs/` (e.g., `my_analysis.py`):

   Simple notebook (only standard fields):
   ```python
   from deriva_ml.execution import notebook_config

   notebook_config(
       "my_analysis",
       defaults={"assets": "my_assets", "datasets": "my_dataset"},
   )
   ```

   Notebook with custom parameters:
   ```python
   from dataclasses import dataclass
   from deriva_ml.execution import BaseConfig, notebook_config

   @dataclass
   class MyAnalysisConfig(BaseConfig):
       threshold: float = 0.5
       num_iterations: int = 100

   notebook_config(
       "my_analysis",
       config_class=MyAnalysisConfig,
       defaults={"assets": "my_assets"},
   )
   ```

2. **Initialize the notebook** (single call does everything):
   ```python
   from deriva_ml.execution import run_notebook

   ml, execution, config = run_notebook("my_analysis")

   # Ready to use:
   # - ml: Connected DerivaML instance
   # - execution: Execution context with downloaded inputs
   # - config: Resolved configuration (config.assets, config.threshold, etc.)

   # At the end of notebook:
   execution.upload_execution_outputs()
   ```

3. **Run notebook with overrides** (command line):
   ```bash
   # Show available configuration options
   uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb --info

   # Run with overrides
   uv run deriva-ml-run-notebook notebooks/my_analysis.ipynb \
     --host dev.eye-ai.org --catalog eye-ai \
     assets=different_assets
   ```

## Tool Preferences

**IMPORTANT: Always start with DerivaML MCP tools for catalog operations.**

When interacting with DerivaML catalogs, **always prefer MCP tools over writing Python scripts**:
- **First**: Use `mcp__deriva-ml__connect_catalog` to establish a connection before any other catalog operation
- Use `mcp__deriva-ml__*` tools for catalog operations (connect, list datasets, download, query, etc.)
- Only write Python scripts when MCP tools are insufficient or when creating production code
- MCP tools provide cleaner, more direct interaction with the catalog

## Key Workflow Rules

- **MUST** commit changes before running models (DerivaML tracks code provenance)
- Use `dry_run=true` during debugging (downloads inputs without creating execution records)
- Tag versions with `bump-version` before significant model runs
- Commit `uv.lock` to repository
- Never commit notebooks with output cells (use `uv run nbstripout --install`)
- Use Google docstring format and type hints
- **Always check function/class signatures before modifying calls** - use `inspect.signature()` or check the source to verify required parameters before editing code that instantiates classes or calls functions

## Eye-AI Dataset Structure

The Eye-AI catalog contains fundus images with glaucoma diagnosis labels.

### Diagnosis Classes
- **No Glaucoma**: Healthy eye
- **Suspected Glaucoma**: Glaucoma indicators present
- **Unknown**: Excluded from binary classification

### Available Datasets
- `test_small` (5-XW4J): Small test dataset for quick debugging
- `subset_10pct_*`: 10% subsets for quick experiments
- `graded_*`: Balanced datasets (800 train / 250 test per class)
- `full_*`: Complete LAC datasets for production training

**For experiments requiring balanced classes, use graded datasets.**

## Overriding Configs at Runtime

```bash
# Choose different configs (no + prefix for groups with defaults)
uv run deriva-ml-run datasets=graded_combined model_config=vgg19_frozen

# Override specific fields (use + for adding new fields)
uv run deriva-ml-run model_config.epochs=50 model_config.learning_rate=0.0001

# Use experiment presets
uv run deriva-ml-run +experiment=vgg19_extended_graded
```
