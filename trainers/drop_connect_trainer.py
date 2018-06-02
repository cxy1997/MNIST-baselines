from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import os
from pydoc import locate

def train(data, model, optimizer, logger, config):
    criterion = locate("torch.nn.%s" % config["criterion"])()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    for epoch in range(config["last_epoch"] + 1, config["epochs"] + 1):
        batch_indices = np.random.choice(data.DATA_SIZE[0], size=config["batch_size"], replace=False)
        inputs = Variable(torch.from_numpy(data.data_train[batch_indices, :]), requires_grad=False)
        targets = Variable(torch.from_numpy(data.label_train[batch_indices]), requires_grad=False)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batches = (data.DATA_SIZE[1] + config["batch_size"] - 1) // config["batch_size"]
        prediction = np.zeros(data.DATA_SIZE[1], dtype=np.uint8)
        for i in range(batches):
            inputs = Variable(torch.from_numpy(data.data_test[i * config["batch_size"]: min((i + 1) * config["batch_size"], data.DATA_SIZE[1]), :]), requires_grad=False)
            with torch.no_grad():
                outputs = model(inputs)
            prediction[i * config["batch_size"]: min((i + 1) * config["batch_size"], data.DATA_SIZE[1])] = np.argmin(outputs.data.cpu().numpy(), axis=1)
        accuracy = accuracy_score(data.label_test, prediction)
        logger.info("Epoch: %d, loss: %0.6f, accuracy: %0.6f" % (epoch, loss.data.cpu().numpy(), accuracy))

        if epoch % config["save_freq"] == 0:
            torch.save(model.state_dict(), os.path.join(config["model_dir"], config["method"], "epoch_%d.pth" % epoch))
            