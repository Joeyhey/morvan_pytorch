import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# make fake data
n_data = torch.ones(100, 2)
# 这里的 x 和 y 不是坐标，横纵坐标都是包含在二维张量 x 中的，y 是 x 的分类标签
# 返回一个张量，包含了从指定均值 means 和标准差 std 的离散正态分布中抽取的一组随机数。
x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
# 将两个张量按维度 0 拼接在一起
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer


# plt.scatter(x.detach()[:, 0], x.detach()[:, 1], c=y.detach(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# method 1
class Net(torch.nn.Module):
    def __init__(self, num_feature, num_hidden, num_output):
        # 继承父类 Net 的所有属性和方法
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(num_feature, num_hidden)
        self.predict = torch.nn.Linear(num_hidden, num_output)

    # 真正搭建的过程在这里
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net1 = Net(2, 10, 2)

# method 2
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)

print(net1)
print(net2)

# 打开 matplotlib 的交互模式使 python 动态显示图片
plt.ion()
plt.show()

# 优化神经网络    随机梯度下降，输入要优化的参数，设定学习效率
optimizer = torch.optim.SGD(net1.parameters(), lr=0.02)

# 开始训练
for t in range(100):
    # 训练 100 步，看每一步的 out，这里的 net(x) 调用了 forward 函数
    out = net2(x)

    # 交叉熵损失函数，计算分类问题，计算误差，真实值在后
    loss = F.cross_entropy(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每学习两步就打印一次
    if t % 2 == 0:
        plt.cla()
        # 返回输入 tensor 中所有元素的最大值，F.softmax(out, dim=1) 对每一行进行 softmax，dim=0 对每一列
        prediction = torch.max(F.softmax(out, dim=1), 1)[1]
        pred_y = prediction.detach().numpy()
        target_y = y.detach().numpy()
        plt.scatter(x.detach()[:, 0], x.detach()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

# 关闭交互模式，回到阻塞模式
plt.ioff()
plt.show()
