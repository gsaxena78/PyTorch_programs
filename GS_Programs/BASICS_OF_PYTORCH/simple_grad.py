import torch
import torch.nn.functional as F
from torch.autograd import grad 

y =  torch.tensor([1.0])
x1 = torch.tensor(0.5)
w1 = torch.tensor([0.6],requires_grad=True)
b =  torch.tensor([0.0], requires_grad=True)
z =  w1 * x1 + b
a =  torch.sigmoid(z)
loss = F.binary_cross_entropy(a,y)
grad_L_w = grad(loss,w1,retain_graph=True)
grad_L_b = grad(loss,b,retain_graph=True)
print(grad_L_w)
print(grad_L_b)
loss.backward()
print(w1.grad)
print(b.grad)