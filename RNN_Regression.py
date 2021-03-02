import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10  # rnn time step
INPUT_SIZE = 1  # rnn input size    不是输入层节点个数，而是说 在一个时间点，RNN 接受一个输入，根据这个输入的 sin 预测 cos
LR = 0.02  # learning rate

# show data
# 用 sin 预测 cos
steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            # 上节用的 LSTM 功能比较强大，但是这里 RNN 就可以满足
            input_size=INPUT_SIZE,  # x 的特征维度
            hidden_size=32,  # rnn hidden unit  不是隐藏层节点个数，是隐藏层的特征维度
            num_layers=1,  # RNN 隐藏层的层数
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)  batch 是输入节点个数，input_size 输入节点大小，这里是一个数
        # h_state (n_layers, batch, hidden_size)    hidden_size 确定了隐层状态 h_state 的维度
        # r_out (batch, time_step, hidden_size)
        # 这里的 h_state 会作为下次的输入进行更新，所以最终只有最后一个状态的 h_state，但是有 batch 个 r_out，所以两个维度不一样
        r_out, h_state = self.rnn(x, h_state)

        # r_out 是经过 hidden_layer 处理的结果，outs 是 nn.Linear 对 r_out 再进行处理最后输出的结果
        # 每一步都经过 Linear 层的加工，所以把每步的产物都放入 list 中作为最终的 RNN + output_layer 输出的东西
        outs = []  # save all predictions
        # 在 pytorch 中可以做一个动态的计算图
        # 对于每一个 time_step，把 r_out 的数据取出来过了 hidden layer，r_out (batch, time_step, hidden_size)，size(1) 代表 time_step
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        # 因为输出是 list 形式，用 stack 压在一起
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state

        # or even simpler, since nn.Linear can accept inputs of any dimension
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None  # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()  # continuously plot
plt.show()

for step in range(100):
    # 截取了一段距离
    start, end = step * np.pi, (step + 1) * np.pi  # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32,
                        endpoint=False)  # float32 for converting torch FloatTensor
    print("\nlen(steps): ", len(steps))
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    # 将输出的数据变维度，变成三维的
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)  # rnn output
    # !! next step is important !!
    h_state = h_state.detach()  # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)  # calculate loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.detach().numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()
