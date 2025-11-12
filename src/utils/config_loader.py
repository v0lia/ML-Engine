import yaml
from pathlib import Path

default_config_path = "src/config/config.yaml"

def get_config(config_path):
    path = Path(config_path).expanduser()
    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)
