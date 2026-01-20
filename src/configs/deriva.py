"""DerivaML Connection Configuration.

This module defines connection configurations for the Deriva catalog.

Configuration Group: deriva_ml
------------------------------
This group specifies which Deriva catalog to connect to. Each configuration
provides connection parameters (hostname, catalog_id, credentials).

REQUIRED: A configuration named "default_deriva" must be defined.
This is used as the default connection when no override is specified.

Example usage:
    # Use default connection
    uv run deriva-ml-run

    # Use a specific connection
    uv run deriva-ml-run deriva_ml=eye-ai-prod
"""

from hydra_zen import store
from deriva_ml import DerivaMLConfig

# ---------------------------------------------------------------------------
# DerivaML Connection Configurations
# ---------------------------------------------------------------------------
# The group name "deriva_ml" must match the parameter name in BaseConfig.

deriva_store = store(group="deriva_ml")

# REQUIRED: default_deriva - used when no connection is specified
# Points to the Eye-AI development catalog
deriva_store(
    DerivaMLConfig,
    name="default_deriva",
    hostname="dev.eye-ai.org",
    catalog_id="eye-ai",
    use_minid=False,
    zen_meta={
        "description": (
            "Eye-AI development catalog (dev.eye-ai.org/eye-ai). "
            "Schema: eye-ai. Contains fundus images for glaucoma classification."
        )
    },
)

# Production Eye-AI catalog (when available)
deriva_store(
    DerivaMLConfig,
    name="eye-ai-prod",
    hostname="www.eye-ai.org",
    catalog_id="eye-ai",
    use_minid=False,
    zen_meta={
        "description": (
            "Eye-AI production catalog (www.eye-ai.org/eye-ai). "
            "Use for production model training and deployment."
        )
    },
)
