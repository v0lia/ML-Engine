# defaults.py

import sys
from pathlib import Path

from torch import nn, optim

default_seed = 42
default_loss_function_class = nn.CrossEntropyLoss
default_optimizer_class = optim.Adam

DATETIME_FORMAT = '%Y%m%d-%H%M%S'
PROJECT_ROOT = Path(sys.path[0]).resolve()

default_config_path = PROJECT_ROOT / "config/config.yaml"
default_models_path = PROJECT_ROOT / "src/models"
default_checkpoints_path = PROJECT_ROOT / "checkpoints"
default_dataset_path = PROJECT_ROOT / "datasets"
default_results_path = PROJECT_ROOT / "results"



