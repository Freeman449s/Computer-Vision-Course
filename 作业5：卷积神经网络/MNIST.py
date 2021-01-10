import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os


class LeNet5(nn.Module):

    def __init__(self):
        """
        构造函数
        """
        super(LeNet5, self).__init__()
        # 卷积层C1 32*32*1 -> 28*28*6
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # padding: 零填充
        # 池化层S2, S4
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层C3 14*14*6 -> 10*10*16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 卷积层C5 5*5*16 -> 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层F6 120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # 输出层 84 -> 10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义一轮计算过程\n
        :param x: 输入图像
        :return: 概率向量
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 将卷积层的每个特征平面展开。16*5*5 -> 16*25。-1表示不确定向量个数，第二个参数表示每个向量的维度
        x = x.view(-1, self.__countDimensions(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __countDimensions(self, x: torch.Tensor) -> int:
        """
        计算矩阵展平成向量后的维度数\n
        :param x: 张量（含有多个尺寸相同的矩阵）
        :return: 矩阵展平成向量后的维度数
        """
        size = x.size()[1:]  # x.size()返回(256, 16, 5, 5)。其中256是batch_size，表示一次训练的图像数
        nDim = 1
        for s in size:
            nDim *= s
        return nDim


def determineDevice():
    """
    判断使用的设备类型，并返回设备对象\n
    :return: 使用的设备对象
    """
    gpuAvailable = torch.cuda.is_available()
    if gpuAvailable:
        print("GPU is available. Will use GPU.")
        device = torch.device("cuda:0")
    else:
        print("GPU is not available. Will use CPU instead.")
        device = torch.device("cpu")
    return device


def train() -> LeNet5:
    """
    训练过程\n
    :return: 经过训练的网络
    """
    device = determineDevice()
    net = LeNet5()
    net.to(device)
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))  # 优化器

    print("Training started.")

    for epoch in range(0, 50):
        net.train()
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()  # 梯度参数归零
            outputs = net(inputs)
            loss = criterion(outputs, labels)  # 计算loss
            loss.backward()  # 反向传播损失来进行优化
            optimizer.step()

            if i % 50 == 0:  # 每训练50批图像输出一次结果
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, i * len(inputs), len(trainLoader.dataset), 100. * i / len(trainLoader), loss.item()))
    # 储存训练数据
    torch.save(net.state_dict(), "State Dict/MNIST.pth")
    print("Training finished.")
    return net


def test(stateDictPath: str) -> None:
    """
    测试过程\n
    :param stateDictPath: 训练所得到数据路径
    :return: 无返回值
    """
    device = determineDevice()

    net = LeNet5()
    net.to(device)
    net.load_state_dict(torch.load(stateDictPath))

    # 定义统计正确率所需变量
    nCorrect = nTotal = 0
    classCorrect = list(0. for i in range(10))  # 创建一个长度为10，元素为0的列表
    classTotal = list(0. for i in range(10))

    print("Test started.")
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            # 预测各张图像的类型。1表示寻找每行的最大值。第一个返回值是每行的最大值，第二个返回值是最大值的索引。
            _, predicted = torch.max(outputs, 1)

            # 统计正确率
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                classCorrect[label] += c[i].item()
                classTotal[label] += 1

            nTotal += labels.size(0)
            nCorrect += (predicted == labels).sum().item()
    # 显示结果
    print("Accuracy : %.2f %%" % (100 * nCorrect / nTotal))
    for i in range(10):
        print("Accuracy of %s : %.2f %%" % (classes[i], 100 * classCorrect[i] / classTotal[i]))


if __name__ == "__main__":
    # 定义一系列图像变换操作
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))]
    )
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')  # 图像所属类别

    trainSet = torchvision.datasets.MNIST(root="Data", train=True, download=False, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=256, shuffle=True, num_workers=2)
    testSet = torchvision.datasets.MNIST(root="Data", train=False, download=False, transform=transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=256, shuffle=False, num_workers=2)

    test("State Dict/MNIST.pth")
