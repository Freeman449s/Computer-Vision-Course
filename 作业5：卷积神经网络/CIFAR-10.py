import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self):
        """
        构造函数
        """
        super(CNN, self).__init__()
        # 卷积层C1 32*32*3 -> 28*28*18
        self.conv1 = nn.Conv2d(3, 18, 5, padding=2)  # 输入3通道，输出18通道，卷积核尺寸5
        # 池化
        self.pool = nn.MaxPool2d(2, 2)
        # 局部响应归一化
        self.norm = nn.LocalResponseNorm(1)
        # 卷积层C4 16*16*18 -> 12*12*36
        self.conv2 = nn.Conv2d(18, 36, 5)
        # 卷积层C7 12*12*36 -> 8*8*72
        self.conv3 = nn.Conv2d(36, 72, 5)
        # 全连接层F10 1*1*72 -> 36
        self.fc1 = nn.Linear(72, 36)
        # 全连接层F11 36 -> 10
        self.fc2 = nn.Linear(36, 10)
        # dropout
        self.dropout = nn.Dropout(0.3)
        # 归一化
        self.softMax = nn.Softmax(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.pool(F.relu(self.conv1(x))))
        x = self.norm(self.pool(F.relu(self.conv2(x))))
        x = self.norm(self.pool(F.relu(self.conv3(x))))
        x = x.view(-1, self.__countDimensions(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def __countDimensions(self, x: torch.Tensor) -> int:
        """
        计算矩阵展平成向量后的维度数\n
        :param x: 张量（含有多个尺寸相同的矩阵）
        :return: 矩阵展平成向量后的维度数
        """
        size = x.size()[1:]
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


def train() -> CNN:
    """
    训练过程\n
    :return: 经过训练的网络
    """
    device = determineDevice()
    net = CNN()
    net.to(device)
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))  # 优化器

    print("Training started.")

    for epoch in range(0, 30):
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
    torch.save(net.state_dict(), "State Dict/CIFAR-10.pth")
    print("Training finished.")
    return net


def test(stateDictPath: str) -> None:
    """
    测试过程\n
    :param stateDictPath: 训练所得到数据路径
    :return: 无返回值
    """
    device = determineDevice()

    net = CNN()
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    classes = ["airplane", 'automobile', "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]  # 图像所属类别

    trainSet = torchvision.datasets.CIFAR10(root="Data/CIFAR", train=True, download=False, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=16, shuffle=True, num_workers=2)
    testSet = torchvision.datasets.CIFAR10(root="Data/CIFAR", train=False, download=False, transform=transform)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=16, shuffle=False, num_workers=2)

    train()
