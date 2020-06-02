import argparse
import datetime
import sys
import numpy
import onnxruntime as ort

import torch

from dataset import LuxorDataset
from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset_path', type=str, default="../l/l2", dest='test_dataset_path')
parser.add_argument('--model_state_dict_path', type=str, default="statedict", dest='model_state_dict_path')
args = parser.parse_args()

test_dataset = LuxorDataset(args.test_dataset_path, 1, 801)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=6, shuffle=False)

ort_session = ort.InferenceSession('luxor.onnx')
input_name = ort_session.get_inputs()[0].name

start_time = datetime.datetime.now()
for i_batch, sample_batched in enumerate(test_loader):
    features = sample_batched["x"]
    outputs = ort_session.run(None, {input_name: features.numpy().astype(numpy.float32)})[0]
    indices = outputs.argmax(1)
    letters = [chr(ord('a') + i) for i in indices]
    print(''.join(letters))
end_time = datetime.datetime.now()
sys.stderr.write("Solving time (sec): %f\n" % (end_time - start_time).total_seconds())
