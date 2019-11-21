import torch

def train(net, data_loader, parameters, device):
    net.to(device=device)
    net.train()

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=parameters.get("lr", 0.001),
        momentum=parameters.get("momentum", 0.0),
        weight_decay=parameters.get("weight_decay", 0.0),
    )
    num_epochs = parameters.get("num_epochs", 5)

    for epoch in range(num_epochs):
        for i_batch, sample_batched in enumerate(data_loader):
            features = sample_batched["x"].to(device=device)
            labels = sample_batched["y"].to(device=device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            print(loss)
            loss.backward()
            if "grad_norm" in parameters and parameters["grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), parameters["grad_norm"])
            optimizer.step()

def evaluate(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_loader):
            features = sample_batched["x"].to(device=device)
            labels = sample_batched["y"].to(device=device)
            labels = labels.to(device=device)
            outputs = net(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
