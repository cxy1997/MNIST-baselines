from __future__ import division, print_function
import os
import argparse
import json
import numpy as np
import torch
from pydoc import locate
from data_loader import MnistLoader
from utils import init_dir, show_config, setup_logger, load_model

parser = argparse.ArgumentParser(description="MNIST classifiers")
parser.add_argument("--method", type=str, default="mlp")

parser.add_argument("--resume", action="store_true")
parser.add_argument("--config-dir", type=str, default="config")
parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument("--model-dir", type=str, default="trained_models")
parser.add_argument("--log-dir", type=str, default="logs")
parser.add_argument("--cuda", type=bool, default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(args.config_dir, "%s.json" % args.method)) as f:
        config = json.load(f)
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    show_config(config)
    
    init_dir(args.model_dir)
    init_dir(args.log_dir)
    init_dir(os.path.join(args.model_dir, args.method))
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_default_tensor_type('torch.FloatTensor')

    data = MnistLoader(flatten=config["flatten"], data_path=args.data_dir, var_per=0.80)
    model = locate("models.%s.%s" % (args.method, config["model_name"]))(in_features=data.data_train.shape[1])
    print('data & model prepared, start to train')
    if args.resume:
        model_path, config["last_epoch"], config["best_accuracy"] = load_model(args.model_dir, args.method)
        if model_path is not None:
            print("Loading latest model from %s" % model_path)
            model.load_state_dict(torch.load(model_path))
    else:
        config["last_epoch"], config["best_accuracy"] = 0, 0.0
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
    model.train()
    optimizer = locate("torch.optim.%s" % config["optimizer"])(model.parameters(), lr = config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    logger = setup_logger(args.method, os.path.join(args.log_dir, "%s.log" % args.method), resume=args.resume)

    train = locate("trainers.%s.train" % config["trainer"])
    train(data, model, optimizer, logger, config)
    