import torch

# Pytorch 中浮点类型的 tensor 才能有梯度
tensor = torch.tensor([[1, 2], [3, 4]], requires_grad=True, dtype=torch.float32)

# 所有 x^2 的平均值
t_out = torch.mean(tensor * tensor)

# print(tensor)
print("\nt_out\n", t_out)

t_out.backward()
# t_out = 1 / 4 * sum(tensor * tensor)
# d(t_out) / d(ten) = 1 / 4 * 2 * tensor = 1/2 * tensor
print("\ntensor.grad\n", tensor.grad)

print("\ntensor.data\n", tensor.data)
print("\ntensor.numpy()\n", tensor.detach().numpy())
