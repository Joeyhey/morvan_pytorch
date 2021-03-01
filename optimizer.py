import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 超参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# unsqueeze 把一维数据变成二维数据
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

# 打印数据，看看长啥样
# plt.scatter(x.numpy(), y.numpy())
# plt.show()

# miniBatch
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# 建立神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 建立 20 个神经元的隐藏层，不是 20 个隐藏层
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# 针对四个不同的优化器建立四个不同的神经网络
net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
# 将他们放在一个 list 中
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# 四个神经网络建立好之后再建立四个不同的优化器，并将其放入 list
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
# 记录四个优化器的损失
losses_his = [[], [], [], []]

for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        # 不同的神经网络使用 for zip 将其捆在一起一个个进行训练
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(batch_x)  # get output for every net
            loss = loss_func(output, batch_y)  # compute loss for every net
            opt.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            opt.step()  # apply gradients
            l_his.append(loss.item())  # loss recoder

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()