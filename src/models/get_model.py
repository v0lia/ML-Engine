import importlib

def get_model_class(model_name: str):
    try:
        module = importlib.import_module(f"src.models.{model_name}")
        return getattr(module, "NeuralNetwork")
    except ModuleNotFoundError:
        raise FileNotFoundError(f"Model file {model_name}.py not found in src/models/")
    except AttributeError:
        raise ValueError(f"Model class in {model_name} must be named NeuralNetwork.")
