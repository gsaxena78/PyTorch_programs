import torch
import torch.nn.functional as F

y = torch.tensor([1.0])
w1 = torch.tensor([0.6])
x1 = torch.tensor(0.5)
b = torch.tensor([0.0])
z = w1 * x1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a,y)
print(loss)