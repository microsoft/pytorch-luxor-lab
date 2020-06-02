import argparse
import math

import torch
from ax.service.managed_loop import optimize

from dataset import LuxorDataset
from model import Net
from utils import train

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default="../l/l2", dest='train_dataset_path')
args = parser.parse_args()

train_dataset = LuxorDataset(args.train_dataset_path, 1, 201)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10*6, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train_wrapper(parameterization):
    net = Net()
    parameterization["num_epochs"] = 1
    last_epoch_loss = train(net=net, data_loader=train_loader, parameters=parameterization, device=device)
    if math.isnan(last_epoch_loss):
        last_epoch_loss = 10**6
    return -last_epoch_loss

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-5, 0.1], "log_scale": True},
        {"name": "grad_norm", "type": "range", "bounds": [1., 100.], "log_scale": True},
    ],
    evaluation_function=train_wrapper
)

print(best_parameters)
