import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层（32*32*1的图像）
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # padding: 零填充
        # 池化层（2*2）
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层（6*14*14的图像）
        self.conv2 = nn.Conv2d(6, 16, 5)
        # LINEAR LAYER(16*5*5-->120)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # LINEAR LAYER(120-->84)
        self.fc2 = nn.Linear(120, 84)
        # LINEAR LAYER(84-->10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 将卷积层的每个特征平面展开。16*5*5->16*25。-1表示不确定向量个数，第二个参数表示每个向量的维度
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
        print('CUDA IS AVAILABLE!')
        device = torch.device("cuda:0")

    net = LeNet5()
    net.to(device)
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))  # 优化器

    for epoch in range(50):  # loop over the dataset multiple times
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

            if i % 50 == 0:  # print every 50 mini-batches
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(trainloader.dataset),
                           100. * i / len(trainloader), loss.item()))
    # 存储训练好的参数
    PATH = './mnist_lenet5.pth'
    torch.save(net.state_dict(), PATH)
    print('Finished Training')
    return net


def test(net):
    # 测试：
    print("Start Testing")
    train_on_gpu = torch.cuda.is_available()
    # gpu是否可用
    if not train_on_gpu:
        print('CUDA IS NOT AVAILABLE!')
        device = torch.device("cpu")
    else:
        print('CUDA IS AVAILABEL!')
        device = torch.device("cuda:0")

    PATH = './mnist_lenet5.pth'
    # 是否直接进行test，无train
    if not net:
        net = LeNet5()
        net.to(device)
        net.load_state_dict(torch.load(PATH))

    # 用于计数
    correct = 0
    total = 0

    # 用于查看不同的类别的正确率分别为多少
    class_correct = list(0. for i in range(10))  # 创建一个长度为10，元素为0的列表
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # 到测试集的images和labels
            outputs = net(images)  # 放入网络中得到output
            # 通过outputs得到预测的类型编号。1表示寻找每行的最大值。第一个返回值是每行的最大值，第二个返回值是最大值的索引。
            _, predicted = torch.max(outputs, 1)

            # 对比结果，如果预测正确，则给相应类的计数器加上一定值，并给总计数器加上一定值
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 输出结果
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':

    transform = transforms.Compose(  # 定义一系列图像变换操作
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])

    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    net = None
    if not os.path.exists('./mnist_lenet5.pth'):
        net = train()
    else:
        print("There's already traning result in the folder,it will start test directly")

    test(net)
