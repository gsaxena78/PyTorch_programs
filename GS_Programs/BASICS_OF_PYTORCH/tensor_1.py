import torch
t0D = torch.tensor(1)
t1D = torch.tensor([1.0,2.0,3.0,4.0,5.0])
t2D = torch.tensor([[1,2,3],[4,5,6]])
t3D = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

print(t2D)
print(t3D)

print(t1D.dtype)