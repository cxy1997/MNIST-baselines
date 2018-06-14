from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import os
import json
from pydoc import locate
from utils import init_dir

def train(data, model, optimizer, logger, config):
    criterion = locate("torch.nn.%s" % config["criterion"])()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    train_batches = (data.DATA_SIZE[0] + config["batch_size"] - 1) // config["batch_size"]
    test_batches = (data.DATA_SIZE[1] + config["batch_size"] - 1) // config["batch_size"]

    for epoch in range(config["last_epoch"] + 1, config["epochs"] + 1):
        if epoch in config["lr_decay_epoch"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= config["lr_decay_rate"]

        loss_sum = 0
        epoch_indices = np.random.permutation(data.DATA_SIZE[0])
        for i in range(test_batches):
            batch_indices = epoch_indices[i * config["batch_size"]: min((i + 1) * config["batch_size"], data.DATA_SIZE[1])]
            inputs = Variable(torch.from_numpy(data.data_train[batch_indices, :]), requires_grad=False).view(-1, 1, 45, 45)
            targets = Variable(torch.from_numpy(data.label_train[batch_indices]), requires_grad=False)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss_sum += loss.data.cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        prediction = np.zeros(data.DATA_SIZE[1], dtype=np.uint8)
        for i in range(test_batches):
            inputs = Variable(torch.from_numpy(data.data_test[i * config["batch_size"]: min((i + 1) * config["batch_size"], data.DATA_SIZE[1]), :]), requires_grad=False).view(-1, 1, 45, 45)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            prediction[i * config["batch_size"]: min((i + 1) * config["batch_size"], data.DATA_SIZE[1])] = np.argmax(outputs.data.cpu().numpy(), axis=1)

        for param in model.parameters():
            param.requires_grad = True
        model.train()

        accuracy = accuracy_score(data.label_test, prediction)
        logger.info("Epoch: %d, loss: %0.6f, accuracy: %0.6f" % (epoch, loss_sum, accuracy))

        if accuracy > config["best_acc"]:
            config["best_acc"] = accuracy
            config["last_epoch"] = epoch
            torch.save(model.state_dict(), os.path.join(config["model_dir"], "%s_model.pth" % config["method"]))
            with open(os.path.join(config["config_dir"], "%s.json" % config["method"]), 'w') as f:
                json.dump(config, f, indent=4)
            
def test(data, model, optimizer, logger, config):
    test_batches = (data.DATA_SIZE[1] + config["batch_size"] - 1) // config["batch_size"]
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    prediction = np.zeros(data.DATA_SIZE[1], dtype=np.uint8)
    for i in range(test_batches):
        inputs = Variable(torch.from_numpy(data.data_test[i * config["batch_size"]: min((i + 1) * config["batch_size"], data.DATA_SIZE[1]), :]), requires_grad=False).view(-1, 1, 45, 45)
        if config["cuda"] and torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = model(inputs)
        prediction[i * config["batch_size"]: min((i + 1) * config["batch_size"], data.DATA_SIZE[1])] = np.argmax(outputs.data.cpu().numpy(), axis=1)

    print('Accuracy: %0.2f' % (100 * accuracy_score(data.label_test, prediction)))
    init_dir(config['output_dir'])
    np.save(os.path.join(config['output_dir'], '%s_pred.npy' % config['method']), prediction)