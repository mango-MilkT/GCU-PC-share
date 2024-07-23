import torch
from torch import nn
from d2l import torch as d2l
import torch_gcu
from .torchgpipe import GPipe

model = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(), nn.Linear(256,10))
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

model.apply(init_weights)

model = GPipe(model, balance=[2, 2], chunks=8)

# 1st partition: nn.Sequential(a, b) on cuda:0
# 2nd partition: nn.Sequential(c, d) on cuda:1

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(model.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

in_device = model.devices[0]
out_device = model.devices[-1]

animator = d2l.Animator(xlabel='epoch', xlim=[1,num_epochs], ylim=[0.3,0.9], legend=['train loss', 'train acc', 'test acc'])
for epoch in range(num_epochs):
    if isinstance(model, torch.nn.Module):
        model.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        X = X.to(in_device, non_blocking=True)
        y = y.to(out_device, non_blocking=True)
        y_hat = model(X)
        l = loss(y_hat, y)
        if isinstance(trainer, torch.optim.Optimizer):
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    train_metrics = metric[0] / metric[2], metric[1] / metric[2]
    
    if isinstance(model, nn.Module):
        model.eval()  # Set the model to evaluation mode
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(in_device, non_blocking=True)
            y = y.to(out_device, non_blocking=True)
            y_hat = model(X)
            metric.add(d2l.accuracy(y_hat, y), y.numel())
    test_acc = metric[0] / metric[1]

    animator.add(epoch+1, train_metrics + (test_acc,))

train_loss, train_acc = train_metrics

assert train_loss < 0.5, train_loss
assert train_acc <= 1 and train_acc > 0.7, train_acc
assert test_acc <= 1 and test_acc > 0.7, test_acc
