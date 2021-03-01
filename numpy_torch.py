import numpy as np
import torch

data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
data_np = np.array(data)

print(
    "matrix",
    "\nnumpy\n", np.matmul(data, data),  # 矩阵相乘
    "\nnumpy dot\n", data_np.dot(data_np),
    "\ntorch\n", torch.mm(tensor, tensor),
    # "\ntorch dot\n", tensor.dot(tensor),    # 报错，点积期望一维张量，但输入的是二维的
)
