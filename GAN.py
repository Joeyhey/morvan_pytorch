import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 为CPU设置种子用于生成随机数，每次得到的随机数是固定的
# torch.manual_seed(1)
# np.random.seed(1)

# hyper parameters
BATCH_SIZE = 64
# 生成器和裁判的学习率
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15  # it could be total point G can draw in the canvas
# 生成从 -1 到 1 的线段，有 15 个点
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


# # show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()


def artist_works():
    # 生成一批优美的画作
    # 生成的时候用一个随机生成，加维度
    a = np.random.uniform(1, 2, BATCH_SIZE)[:, np.newaxis]
    # 曲线    根据 PAINT_POINTS 点生成一元二次函数
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings


# 学习著名画家，产生和他类似的数据
Generator = nn.Sequential(
    # 新手画家，输入的是一些随机灵感
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    # 随机生成 15 个 y 轴的点
    nn.Linear(128, ART_COMPONENTS),
)

Discriminator = nn.Sequential(
    # 新手鉴赏家，可以接受一幅画（里面的 15 个点）
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    # 输出是一个判别，判别是著名画家还是新手画家
    nn.Linear(128, 1),
    # 结果用百分比表示
    nn.Sigmoid()
)

opt_D = torch.optim.Adam(Discriminator.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(Generator.parameters(), lr=LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_works()
    # 新手画家产生的 idea 是随机产生的，通过随即灵感来画画，生成的形状是 BATCH_SIZE ，用来训练，
    # 一个 batch 里面有 N_IDEAS 个数据，纵坐标是 batch，横坐标 ideas
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
    # 用 idea 创造画
    G_paintings = Generator(G_ideas)
    prob_artist_g = Discriminator(G_paintings)
    # 生成器的概率，增加新手画家的概率
    G_loss = torch.mean(torch.log(1. - prob_artist_g))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    # 有多少概率是著名画家的画
    prob_artist_a = Discriminator(artist_paintings)
    prob_artist_g = Discriminator(G_paintings.detach())

    # 鉴赏家的损失函数，目标是最大化 log(d) + log(1-g)，因为 pytorch 只有最小值，所以加负号
    # 取平均值，增加识别著名画家的画的概率，减少被认为是著名画家但实际是新手画家的画
    D_loss = -torch.mean(torch.log(prob_artist_a) + torch.log(1. - prob_artist_g))

    opt_D.zero_grad()
    # 反向传播之后要保留网络中的一些参数，给下一次 G 里面的反向传播
    D_loss.backward(retain_graph=True)
    opt_D.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist_a.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01)

    plt.ioff()
    plt.show()
