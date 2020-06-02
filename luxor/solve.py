import argparse
import datetime
import sys

import torch

from dataset import LuxorDataset
from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset_path', type=str, default="../l/l2", dest='test_dataset_path')
parser.add_argument('--model_state_dict_path', type=str, default="statedict", dest='model_state_dict_path')
args = parser.parse_args()

test_dataset = LuxorDataset(args.test_dataset_path, 1, 801)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=6, shuffle=False)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = Net()
model = torch.jit.script(model)
model.to(device)
model.load_state_dict(torch.load(args.model_state_dict_path))
model.eval()

start_time = datetime.datetime.now()
with torch.no_grad():
    for i_batch, sample_batched in enumerate(test_loader):
        features = sample_batched["x"].to(device=device)
        outputs = model(features)
        indices = outputs.argmax(1)
        letters = [chr(ord('a') + i) for i in indices]
        print(''.join(letters))
end_time = datetime.datetime.now()
sys.stderr.write("Solving time (sec): %f\n" % (end_time - start_time).total_seconds())
