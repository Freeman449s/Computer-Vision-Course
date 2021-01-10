import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim as optim
import os


# 定义卷积神经网络结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层（32*32*3的图像）
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 卷积层（16*16*16）
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 卷积层（8*8*32）
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # LINEAR LAYER(64*4*4-->500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear层（500，10）
        self.fc2 = nn.Linear(500, 10)
        # dropout(p=0.3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # 平面化向量
        x = x.view(-1, 64 * 4 * 4)
        # 增加了dropout层，防止过拟合
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train():
    print("Start Training")
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA IS NOT AVAILABLE!')
        device = torch.device("cpu")
    else:
        print('CUDA IS AVAILABEL!')
        device = torch.device("cuda:0")

    net = CNN()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
    for epoch in range(30):
        net.train()
        for i, data in enumerate(trainloader, 0):
            # 获得输入labels和inputs
            inputs, labels = data[0].to(device), data[1].to(device)
            # 将梯度参数归零
            optimizer.zero_grad()
            # 将输入放入网络中，获得output
            outputs = net(inputs)
            # 通过output和labels利用损失函数来计算loss
            loss = criterion(outputs, labels)
            loss.backward()  # loss反向传播
            optimizer.step()

            if i % 50 == 0:  # print every 2000 mini-batches
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(trainloader.dataset),
                           100. * i / len(trainloader), loss.item()))

    print('Finished Training')
    PATH = './cifar_cnn.pth'
    torch.save(net.state_dict(), PATH)


def test(net):
    print("Start Testing")
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA IS NOT AVAILABLE!')
        device = torch.device("cpu")
    else:
        print('CUDA IS AVAILABEL!')
        device = torch.device("cuda:0")
    # 测试：
    # 是否直接进行test，无train
    PATH = 'cifar_cnn.pth'
    if not net:
        net = CNN()
        net.to(device)
        net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':

    # 将数据转换为torch.FloatTensor,并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    net = None
    if not os.path.exists('./cifar_cnn.pth'):
        net = train()
    else:
        print("There's already traning result in the folder,it will start test directly")

    test(net)
