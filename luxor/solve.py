import os

import torch
from model import Net
from dataset import LuxorDataset

train_dataset = LuxorDataset("l/l2", 1, 201)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10*6, shuffle=False)


model = Net()
model = model.cuda()
model.train()

#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    for i_batch, sample_batched in enumerate(train_loader):
        features = sample_batched["x"].cuda()
        labels = sample_batched["y"].cuda()
        optimizer.zero_grad()
        outputs = model(features)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        #print(loss)
        loss.backward()
        optimizer.step()


test_dataset = LuxorDataset("l/l2", 1, 801)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=6, shuffle=False)

model.eval()
with torch.no_grad():
    for i_batch, sample_batched in enumerate(test_loader):
        features = sample_batched["x"].cuda()
        outputs = model(features)
        indices = outputs.argmax(1)
        letters = [chr(ord('a') + i) for i in indices]
        print(''.join(letters))
