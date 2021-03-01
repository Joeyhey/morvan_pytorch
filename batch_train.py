import torch
import torch.utils.data as Data

BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 用 torch 创建一个数据库，data_tensor=x, target_tensor=y
torch_dataset = Data.TensorDataset(x, y)
# 用 load 使训练数据变成小批量，shuffle 表示是否训练时随机打乱数据，这里打乱。
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 迭代完所有的训练数据 1 次，称为一个epoch，迭代 3 次
for epoch in range(3):
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标。
    for step, (batch_x, batch_y) in enumerate(loader):
        # 训练
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())
