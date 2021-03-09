import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate
# 是否下载好数据，需要下载为 True，下好以后改成False
DOWNLOAD_MNIST = False

# 去官网下载 MNIST 数据集，保存在 root 路径，作为 training data，
# 通过 transforms 将原始数据改变成需要的形式，原始数据会从（0，255）压缩到（0，1）
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# print(train_data.data.size())
# print(train_data.targets.size())
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()


# 划分数据
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 获取 test data，train 设置为 False
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
# 使一维数据变成二维，后面除以 255 是因为上面的 test_data 实际还是（0，255），加点是浮点型
# !!!!!!!!gpu 加速!!!!!!!!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].to(device) / 255.
# 只取前两千个，节省时间
test_y = test_data.targets[:2000].to(device)


# 构建 CNN 网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # 包含卷积层（过滤器，filter，Conv2d 就是一个信息过滤器，在图片上移动收集信息，filter 的高度用来提取出卷积出来的
            # 特征属性，高度即提取出来的特征值个数，这就是 Conv2d 做的事情。
            nn.Conv2d(  # 图片是（1，28，28）的维度，（图片高度，宽，高）
                in_channels=1,  # 输入图片的高度
                out_channels=16,  # filter 的个数
                kernel_size=5,  # 卷积核尺寸()
                stride=1,  # 卷积步长
                padding=2  # 填充操作，填充值为0，如果 stride=1：padding = (kernel_size - 1) / 2
            ),  # -> (16, 28, 28)
            nn.ReLU(),  # -> (16, 28, 28)
            # 经过 filter 后，图片已经变成一个更厚的图片，MaxPool2d 筛选局域内最大的值作为它的特征
            nn.MaxPool2d(kernel_size=2),  # -> (16, 14, 14)，因为 filter 每次可以扫描到 2 但只选取一个点
        )
        # 创建第二个池化层
        self.conv2 = nn.Sequential(  # -> (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # -> (32, 14, 14)
            nn.ReLU(),  # -> (32, 14, 14)
            nn.MaxPool2d(kernel_size=2),  # -> (32, 7, 7)
        )
        # 输出，三维数据展平，将三维数据展平成二维数据，有 10 个分类
        self.out = nn.Linear(32 * 7 * 7, 10)

    # 展平操作在 forward 中进行
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        # 进行扩展展平操作，-1 的过程就是将数据变到一起
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)

        output = self.out(x)
        return output


cnn = CNN()
# print(cnn)
# !!!!!!!!gpu 加速!!!!!!!!
cnn.to(device)

# 训练过程
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    # 只有在这个 loop 时才会将 data 压缩成（0，1）
    for step, (x, y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        # !!!!!!!!gpu 加速!!!!!!!!
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b_x = x.to(device)
        b_y = y.to(device)
        output = cnn(b_x)  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            test_output = cnn(test_x)
            # !!!!!!!!gpu 加速!!!!!!!!
            pred_y = torch.max(test_output, 1)[1].to(device).detach().numpy()
            accuracy = float((pred_y == test_y.detach().numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.detach().numpy(), '| test accuracy: %.2f' % accuracy)
            # if HAS_SK:
            #     # Visualization of trained flatten layer (T-SNE)
            #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            #     plot_only = 500
            #     low_dim_embs = tsne.fit_transform(last_layer.detach().numpy()[:plot_only, :])
            #     labels = test_y.numpy()[:plot_only]
            #     plot_with_labels(low_dim_embs, labels)

# test data 放前十个数据进去，，再输出预测的数据 pred_y，进行比较
test_output = cnn(test_x[:10])
# !!!!!!!!gpu 加速!!!!!!!!
pred_y = torch.max(test_output, 1)[1].to(device).detach().numpy()
pred_y = pred_y.cpu()

print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
