import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import time

import torch_gcu

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

device = torch_gcu.gcu_device()

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
    plt.savefig(f"./img/test_result_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}.jpg")
    plt.show()
    return axes

n = 6
X, y = next(iter(test_iter))
X, y = X.to(device), y.to(device)
trues = get_fashion_mnist_labels(y)
titles = trues
show_images(
        X[0:n].reshape((n, resize, resize)).cpu(), 1, n, titles=titles[0:n])
