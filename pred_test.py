import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import logging

import torch_gcu

from utils.Timer import Timer
# 0.日志
filename = f"./logger/{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.txt"
logger = logging.getLogger('VGG')
logger.setLevel(logging.INFO)

# 创建文件处理器，并设置级别为INFO
file_handler = logging.FileHandler(filename)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加文件处理器到logger
logger.addHandler(file_handler)


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
# device = torch.device('cpu')
device_info = f'training on {device}'
print(device_info)
logger.info(device_info)
net.to(device=device)

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

timer = Timer()

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
    plt.tight_layout()
    plt.savefig(f"./img/test_result_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.jpg")
    plt.show()
    return axes

timer.start()
net.eval()
n = 6
X, y = next(iter(test_iter))
X, y = X.to(device), y.to(device)
trues = get_fashion_mnist_labels(y)
preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
timer.stop()
test_info = f'test time cost {timer.times[-1]:.3f}'
print(test_info)
input("pred finished!")
show_images(
        X[0:n].reshape((n, resize, resize)).cpu(), 1, n, titles=titles[0:n])

timer.stop()
test_info = f'test time cost {timer.times[-1]:.3f}'
print(test_info)
logger.info(test_info)

