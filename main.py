from __future__ import division, print_function
import os
import argparse
import json
from pydoc import locate
from data_loader import MnistLoader
from utils import show_config

parser = argparse.ArgumentParser(description="MNIST classifiers")
parser.add_argument("--method", type=str, default="drop_connect")

parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument("--model-dir", type=str, default="trained_models")

if __name__ == "__main__":
    args = parser.parse_args()
    config_path = os.path.join("config", "%s.json" % args.method)
    with open(config_path) as f:
        config = json.load(f)
    show_config(config)

    data = MnistLoader(flatten=config["flatten"], data_path=args.data_dir)
    model = locate("models.%s.%s" % (args.method, config["model_name"]))()
    trainer = locate("trainers.%s.trainer" % args.method)(data, model, config)
    trainer.train()
    