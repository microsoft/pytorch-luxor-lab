import argparse
import datetime

import torch

from dataset import LuxorDataset
from model import Net
from utils import evaluate, train

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default="../l/l2", dest='train_dataset_path')
parser.add_argument('--eval_dataset_path', type=str, default="../l/l2", dest='eval_dataset_path')

parser.add_argument('--lr', type=float, default=0.0001, dest='lr')
parser.add_argument('--grad_norm', type=float, default=0, dest='grad_norm')
parser.add_argument('--num_epochs', type=int, default=10, dest='num_epochs')
parser.add_argument('--model_state_dict_path', type=str, default="statedict", dest='model_state_dict_path')

args = parser.parse_args()

train_dataset = LuxorDataset(args.train_dataset_path, 1, 190)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10*6, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net = Net()
net = torch.jit.script(net)

start_time = datetime.datetime.now()
last_epoch_loss = train(net, train_loader, {"num_epochs": args.num_epochs, "lr": args.lr, "grad_norm": args.grad_norm}, device)
end_time = datetime.datetime.now()

print("Training time (sec):", (end_time - start_time).total_seconds())
print("Last epoch loss:", last_epoch_loss)

eval_dataset = LuxorDataset(args.eval_dataset_path, 190, 201)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=10*6, shuffle=False)

torch.save(net.state_dict(), args.model_state_dict_path)

score = evaluate(net, eval_loader, device)
print("Validation accuracy:", score)
