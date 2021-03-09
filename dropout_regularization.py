import matplotlib.pyplot as plt
import torch

# torch.manual_seed(1)    # reproducible

N_SAMPLES = 20
N_HIDDEN = 300

# 制造数据
# training data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# test data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# show data 显示制造的数据
plt.scatter(x.detach().numpy(), y.detach().numpy(), c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x.detach().numpy(), test_y.detach().numpy(), c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# 会造成过拟合的网络，因为输入数据太少神经元又太多
net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

# dropout 正则化，在激活层之前插入，随机抽取 50% 的神经元节点数屏蔽掉
net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

# 打印两个神经网络的不同之处
print(net_overfitting)  # net architecture
print(net_dropped)

# 优化两个神经网络
optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

plt.ion()  # something about plotting

# 训练
for t in range(500):
    pred_ofit = net_overfitting(x)
    pred_drop = net_dropped(x)
    loss_ofit = loss_func(pred_ofit, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t % 10 == 0:
        # 对于 dropout 的功能，当在想要测试时，即在预测时需要把 dropout 功能取消，即不屏蔽 50% 数据
        # change to eval mode in order to fix drop out effect
        net_overfitting.eval()  # 变成预测模式   前面都是net_overfitting.train() train模式
        net_dropped.eval()  # parameters for dropout differ from train mode

        # plotting
        plt.cla()

        # 得到预测结果
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)

        plt.scatter(x.detach().numpy(), y.detach().numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.detach().numpy(), test_y.detach().numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.detach().numpy(), test_pred_ofit.detach().numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.detach().numpy(), test_pred_drop.detach().numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).detach().numpy(),
                 fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).detach().numpy(),
                 fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left');
        plt.ylim((-2.5, 2.5));
        plt.pause(0.1)

        # change back to train mode
        # 预测完之后又要把神经网络放回去训练，再变回训练模式
        net_overfitting.train()
        net_dropped.train()

plt.ioff()
plt.show()
