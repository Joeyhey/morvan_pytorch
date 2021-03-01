import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fake data
# 在 torch 中数据是有维度的，unsqueeze 把一维数据变成二维数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# y = x^2 + 一些噪点的影响
y = x.pow(2) + 0.2 * torch.rand(x.size())


# plt.scatter(x.detach(), y.detach())
# plt.show()


# 继承了 torch 里面的 Module 的模块，Neural Network 的很多功能都包含在内
class Net(torch.nn.Module):
    # 初始化神经网络，需要包含多少个特征、隐藏层的神经元、输出，是个数
    def __init__(self, num_feature, num_hidden, num_output):
        # 继承父类 Net 的所有属性和方法
        super(Net, self).__init__()
        # 神经网络的层信息都是模块当中的一个属性，所以这就是一层隐藏层
        self.hidden = torch.nn.Linear(num_feature, num_hidden)
        # 输入是隐藏层的 n_hidden 个的输入信息，因为只是预测一个 y，所以实际上只输出一个数值，为 1，将他包含到 output 中
        self.predict = torch.nn.Linear(num_hidden, num_output)

    # 真正搭建的过程在这里
    def forward(self, x):
        # 首先在隐藏层加工输入信息，然后用激励函数激活一下信息，
        x = F.relu(self.hidden(x))
        # 在预测时不用激励函数，因为预测值大部分在 -∞ 到 +∞，用了就不对了
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
print(net)

# 打开 matplotlib 的交互模式使 python 动态显示图片
plt.ion()
plt.show()

# 优化神经网络    随机梯度下降，输入要优化的参数，设定学习效率
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# 开始训练
for t in range(100):
    # 训练 100 步，直接看每一步的 prediction，这里的 net(x) 调用了 forward 函数
    prediction = net(x)

    # 均方损失函数，处理回归问题足够了，计算误差，真实值在后
    loss = F.mse_loss(prediction, y)

    # optimizer 要优化神经网络里的所有参数，因为 pytorch 计算梯度是累加的，
    # 为了方便 RNN 的计算，这里不是 RNN，所以要将他们的梯度都清零，再反向传播
    optimizer.zero_grad()
    # 反向传播，计算节点中的梯度
    loss.backward()
    # 得到梯度，进行优化
    optimizer.step()

    # 每学习五步就打印一次
    if t % 5 == 0:
        # 清除轴，当前活动轴在当前图中。 它保持其他轴不变
        plt.cla()
        # 用于产生一个离散的散点图
        plt.scatter(x.detach(), y.detach())
        # 线图，x轴、y轴、红色实线线的宽度
        plt.plot(x.detach(), prediction.detach(), 'r-', lw=5)
        # 图上坐标轴为（0.5，0）点上的说明
        plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
        # 暂停功能，如果有活动图形，它将在暂停之前进行更新和显示
        plt.pause(0.1)

# 关闭交互模式，回到阻塞模式，否则无法 show
plt.ioff()
plt.show()

