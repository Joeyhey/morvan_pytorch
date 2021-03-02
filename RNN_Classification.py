import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28  # rnn time step / image height  输入 28 次
INPUT_SIZE = 28  # rnn input size / image width  每次输入 28 个像素点
LR = 0.01  # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data

train_data = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# plot one example
print(train_data.data.size())  # (60000, 28, 28)
print(train_data.targets.size())  # (60000)
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()

test_data = dsets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
test_x = test_data.data.type(torch.FloatTensor)[: 2000] / 255.
test_y = test_data.targets.numpy().squeeze()[: 2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True  # input & output will has batch size as 1st dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)  分线程（临时细胞状态）的 hidden_state
        # h_c shape (n_layers, batch, hidden_size)  主线程（当前细胞状态）的 hidden_state
        # None 表示没有第一个 hidden_state，如果有就是它的数据
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        # -1 代表这个位置维度大小未定,由其他位置的数字来推断。但根据算法该值应与batch_size保持一致，
        # 但由于可能出现多余情况故以-1保持算法稳定
        b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_out = rnn(test_x)
            pred_y = torch.max(test_out, 1)[1].detach().numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

test_output = rnn(test_x[: 10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].detach().numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
