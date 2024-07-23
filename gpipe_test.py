import torch
from torch import nn
from d2l import torch as d2l
import torch_gcu
import torchgpipe

model = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(), nn.Linear(256,10))
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

model.apply(init_weights)

model = torchgpipe.GPipe(model, balance=[2, 2], chunks=8)

# 1st partition: nn.Sequential(a, b) on cuda:0
# 2nd partition: nn.Sequential(c, d) on cuda:1

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(model.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
input("data loaded!")

in_device = model.devices[0]
out_device = model.devices[-1]
input("in & out device set!")

animator = d2l.Animator(xlabel='epoch', xlim=[1,num_epochs], ylim=[0.3,0.9], legend=['train loss', 'train acc', 'test acc'])
for epoch in range(num_epochs):
    if isinstance(model, torch.nn.Module):
        model.train()
        input("train mode on!")
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        X = X.to(in_device, non_blocking=True)
        y = y.to(out_device, non_blocking=True)
        input("to device complete!")
        y_hat = model(X)
        input("forward compute complete!")
        l = loss(y_hat, y)
        if isinstance(trainer, torch.optim.Optimizer):
            trainer.zero_grad()
            input("zero grad complete!")
            l.mean().backward()
            input("backward compute complete!")
            trainer.step()
            input("optim step complete!")
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    train_metrics = metric[0] / metric[2], metric[1] / metric[2]
    
    if isinstance(model, nn.Module):
        model.eval()  # Set the model to evaluation mode
        input("eval mode on!")
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

input("all complete!")
