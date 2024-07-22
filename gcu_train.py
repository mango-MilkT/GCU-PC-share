import sys
import os
import time
import torch
import logging
import torch_gcu
import numpy as np
import torch.nn as nn
from torch.optim import Adam,SGD
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,CIFAR100
from torch.optim.lr_scheduler import CosineAnnealingLR

log_file = os.path.splitext(os.path.basename(__file__))[0] + ".log"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    ,datefmt="%Y-%m-%d %H:%M:%S"
    ,level=logging.DEBUG
    ,filename=log_file
    ,filemode="w"
)
logger = logging.getLogger(__name__)

#logger.add(log_file, enqueue=True)

# 设置随机种子
seed_value = 40
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# 参数设置
num_epochs = 200
lr = 0.01
batch_size = 32
num_workers = 0
device = torch_gcu.gcu_device()
logger.info(f">>> [init]seed_value:{seed_value}, device:{device}, num_epochs:{num_epochs}, lr:{lr}, batch_size:{batch_size}, num_workers:{num_workers}, log_file:{log_file}")

# normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
normalize = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# 加载训练集和验证集
trainset = CIFAR100(
    root='./datas', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
logger.info(">>> [datasets]success to load train datasets")

testset = CIFAR100(
    root='./datas', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
logger.info(">>> [datasets]success to load test datasets")

model = resnet50(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=lr)
optimizer = SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    train_st = time.time()
    model.train()
    epoch_loss = []
    corrent = 0.0
    for idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch_gcu.optimizer_step(optimizer, [loss, outputs], model=model)
        optimizer.zero_grad()
        epoch_loss.append(loss.item())
        outputs_argmax = torch.argmax(outputs,dim=1)
        corrent += targets.eq(outputs_argmax.view_as(targets)).sum()
        # if idx % 100 == 0:
        #     logger.info(f">>> [train]epoch:{epoch+1:02}/{num_epochs}, step:{idx+1:03}/{len(trainloader)}, loss:{loss.item():.2f}")
    acc = corrent/len(trainloader.dataset) * 100
    train_et = time.time()
    logger.info(f">>> [train]epoch:{epoch+1:03}/{num_epochs}, current mean loss:{torch.Tensor(epoch_loss).mean().item():.4f}, calc acc:{acc:.2f}%")
    logger.info(f">>> [time]epoch:{epoch+1:03}/{num_epochs} train mean cost time:{round(train_et - train_st)}(s)")

def test(epoch):
    test_st = time.time()
    model.eval()
    corrent = 0.0
    with torch.no_grad():
        for idx,(inputs,targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            outputs_argmax = torch.argmax(outputs,dim=1)
            corrent += targets.eq(outputs_argmax.view_as(targets)).sum()
            if idx % 10 == 0:
                logger.info(f">>> [test]epoch:{epoch+1:03}/{num_epochs}, step:{idx:03}/{len(testloader)}, loss:{loss.item():.2f}, corrent:{corrent}")
    acc = corrent/len(testloader.dataset) * 100
    test_et = time.time()
    logger.info(f">>> [test]epoch:{epoch+1:03}/{num_epochs} calc acc:{acc:.2f}%")
    logger.info(f">>> [time]epoch:{epoch+1:03}/{num_epochs} test mean cost time:{round(test_et-test_st)}(s)")

for epoch in range(num_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
