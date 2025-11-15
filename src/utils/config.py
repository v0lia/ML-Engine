# config.py

import yaml
import shutil
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.defaults import default_config_path

def open_config_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_config(config_path):
    logger = get_logger()
    path = Path(config_path).expanduser()
    logger.info(f"Loading config: {path}")

    try:
        return open_config_file(path), path
    except FileNotFoundError:
        logger.warning(f'No config "{path}" found.\nLoading default: {default_config_path}')
        try:
            return open_config_file(default_config_path), default_config_path
        except Exception as e:
            logger.critical(f"Failed to load default config: {e}")
            raise

def copy_config_to_run_dir(config_path, run_dir):
    run_dir = Path(run_dir).resolve()
    dst = run_dir / "config" / config_path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, dst)
    return dst