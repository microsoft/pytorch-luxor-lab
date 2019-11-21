import argparse
import os

import torch

from dataset import LuxorDataset
from model import Net
from utils import evaluate, train

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default="../l/l2", dest='train_dataset_path')
parser.add_argument('--eval_dataset_path', type=str, default="../l/l2", dest='eval_dataset_path')

parser.add_argument('--learning_rate', type=float, default=0.001, dest='learning_rate')
parser.add_argument('--grad_norm', type=float, default=0, dest='grad_norm')
parser.add_argument('--num_epochs', type=int, default=100, dest='num_epochs')

args = parser.parse_args()

train_dataset = LuxorDataset(args.train_dataset_path, 1, 101)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10*6, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net = Net()

train(net, train_loader, {"num_epochs": args.num_epochs, "lr": args.learning_rate, "grad_norm": args.grad_norm}, device)

eval_dataset = LuxorDataset(args.eval_dataset_path, 101, 201)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=10*6, shuffle=False)

score = evaluate(net, eval_loader, device)
print(score)
