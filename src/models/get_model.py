# get_model.py

import importlib
# from pathlib import Path

from src.utils.logger import get_logger
from src.utils.defaults import PROJECT_ROOT, default_models_path

def get_model(model_class_name):
    logger = get_logger()

    models_dir = default_models_path.resolve()

    matched_file = None
    for file in models_dir.glob("*.py"):
        if file.stem.lower() == model_class_name.lower():
            matched_file = file
            break

    if matched_file is None:
        error_text = f'Model file "{model_class_name}.py" not found in {models_dir}'
        logger.error(error_text)
        raise FileNotFoundError(error_text)
    
    relative_path = matched_file.relative_to(PROJECT_ROOT)
    module_name = ".".join(relative_path.with_suffix("").parts)
    
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        error_text = f'Unable to import module "{module_name}"'
        logger.error(error_text + f":\n{e}")
        raise FileNotFoundError(error_text) from e

    try:
        model_class = getattr(module, model_class_name)
    except AttributeError as e:
        error_text = f'Not found "{model_class_name}" class in module "{module_name}"'
        logger.error(error_text + f":\n{e}")
        raise ValueError(error_text)

    logger.info(f"Model: {model_class_name}")
    return model_class()
