import argparse

def parse_args(default_config_path):
    parser = argparse.ArgumentParser(description="Neural network train/eval pipeline")
    parser.add_argument("mode", choices=["train", "eval", "evaluate"],
                        help="Run mode: train or eval")
    parser.add_argument("model",
                        help="Name of a model from src/models/, e.g. CNN_v1")
    parser.add_argument("-c", "--config", "--cfg", default=default_config_path,
                        help=f"Path to YAML config file; default: {default_config_path}")
    return parser.parse_args()
