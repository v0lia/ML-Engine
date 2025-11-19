# get_run_dir.py

from pathlib import Path
from datetime import datetime

from src.utils.defaults import DATETIME_FORMAT, default_results_path

def get_run_dir(mode: str, model_name: str):
    folder_name = mode + "_" + model_name + "_" + f"{datetime.now():{DATETIME_FORMAT}}"
    run_dir_path = Path(default_results_path / folder_name).resolve()
    Path(run_dir_path).mkdir(parents=True, exist_ok=True)
    return run_dir_path