import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200)  # 从 -5 到 5 的区间内取 200 个点数据
x_np = x.detach().numpy()   # 画图时 torch 的数据格式不能被 matplotlib 识别，需要转换为 numpy

# 做一些激励函数的激活值
y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
# softmax 也是激励函数，但不是用来做线图，而是用来做概率图
y_softplus = F.softplus(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()