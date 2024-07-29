import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_gcu

from utils.Timer import Timer

# 1.预处理操作
trans = []
resize = 224
trans.append(transforms.Resize(resize))
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)

# 2.数据集读取和迭代器生成
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=False)

batch_size = 128
train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=False)

# 3.定义模型
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

conv_arch = ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))
net = vgg(conv_arch)

# 4.指定设备
device = torch_gcu.gcu_device()
print('training on', device)
net.to(device=device)

# 5.参数初始化
def init_params(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_params)

# 6.定义损失函数(交叉熵)
criterion = nn.CrossEntropyLoss()

# 7.定义优化器
num_epochs = 10
lr = 0.05
updater = torch.optim.SGD(net.parameters(), lr=lr) # 要在指定设备之后, 因为这里用到了parameters
scheduler = CosineAnnealingLR(updater, T_max=num_epochs)

# *8.精度计算
def accuracy(y_hat, y):
    """
    计算预测正确的数量

    y_hat: the direct outputs(before softmax, actually doesn't matter) of net \n
            (batch_size, output_dim) \n
    y: the labels \n
        (batch_size, 1)
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    compare = y_hat.type(y.dtype) == y
    return float(compare.type(y.dtype).sum())

# 9.正式迭代训练
all_train_loss = []
all_train_acc = []
all_test_acc = []
all_num_train_examples = 0
timer = Timer()
for epoch in range(num_epochs):
    epoch_train_loss, epoch_train_acc, epoch_test_acc = 0., 0., 0.
    num_train_examples = 0
    net.train()
    timer.start()
    for i, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)
        # print(X,y)
        y_hat = net(X)
        loss = criterion(y_hat, y)
        updater.zero_grad()
        loss.backward()
        torch_gcu.optimizer_step(updater, [loss, y_hat], model=net)
        with torch.no_grad():
            epoch_train_loss += float(loss * y.numel())
            epoch_train_acc += accuracy(y_hat, y)
            num_train_examples += y.numel()
    epoch_train_loss /= num_train_examples
    epoch_train_acc /= num_train_examples

    all_train_loss.append(epoch_train_loss)
    all_train_acc.append(epoch_train_acc)
    all_num_train_examples += num_train_examples
    timer.stop()
    print(f'epoch {epoch+1}, loss {epoch_train_loss:.3f}, acc {epoch_train_acc:.3f}, time cost {timer.times[-1]:.3f}')
    # input('first epoch')

    num_test_examples = 0
    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            epoch_test_acc += accuracy(y_hat, y)
            num_test_examples += y.numel()
        epoch_test_acc /= num_test_examples
        all_test_acc.append(epoch_test_acc)
    
    scheduler.step()

print(f'loss {epoch_train_loss:.3f}, train acc {epoch_train_acc:.3f}, '
        f'test acc {epoch_test_acc:.3f}')
print(f'{num_train_examples / timer.sum():.1f} examples/sec '
        f'on {str(device)}')

plt.plot(range(1, num_epochs+1), all_train_loss, label='train loss')
plt.plot(range(1, num_epochs+1), all_train_acc, linestyle='--', label='train acc')
plt.plot(range(1, num_epochs+1), all_test_acc, linestyle='--', label='test acc')
plt.title('')
plt.xlabel('epoch')
plt.xlim([1, num_epochs])
plt.legend()
plt.show()



# *.可视化
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes

n = 6
X, y = next(iter(test_iter))
X, y = X.to(device), y.to(device)
trues = get_fashion_mnist_labels(y)
preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
show_images(
        X[0:n].reshape((n, resize, resize)).cpu(), 1, n, titles=titles[0:n])