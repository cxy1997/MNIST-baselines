from __future__ import division, print_function
import os
import argparse
import json
import numpy as np
import torch
from pydoc import locate
from data_loader import MnistLoader
from utils import init_dir, show_config, setup_logger, latest_model

parser = argparse.ArgumentParser(description="MNIST classifiers")
parser.add_argument("--method", type=str, default="drop_connect")

parser.add_argument("--resume", action="store_true")
parser.add_argument("--config-dir", type=str, default="config")
parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument("--model-dir", type=str, default="trained_models")
parser.add_argument("--log-dir", type=str, default="logs")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(args.config_dir, "%s.json" % args.method)) as f:
        config = json.load(f)
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    show_config(config)
    config["last_epoch"] = 0
    
    init_dir(args.model_dir)
    init_dir(args.log_dir)
    init_dir(os.path.join(args.model_dir, args.method))
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_default_tensor_type('torch.FloatTensor')

    data = MnistLoader(flatten=config["flatten"], data_path=args.data_dir)
    model = locate("models.%s.%s" % (args.method, config["model_name"]))()
    if args.resume:
        model_path, config["last_epoch"] = latest_model(args.model_dir, args.method)
        print("Loading latest model from %s" % model_path)
        model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    optimizer = locate("torch.optim.%s" % config["optimizer"])(model.parameters(), lr = config["lr"])
    logger = setup_logger(args.method, os.path.join(args.log_dir, "%s.log" % args.method))

    train = locate("trainers.%s_trainer.train" % args.method)
    train(data, model, optimizer, logger, config)
    