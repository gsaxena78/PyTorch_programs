import torch
import numpy as np

# Scalar
t0D = torch.tensor(1)

# 1-D with datatype specified
t1D = torch.tensor([1.0,2.0,3.0,4.0,5.0],dtype=torch.float)

# 2-D fully specified
t2D = torch.tensor([[1,2,3],[4,5,6]])

# 2-D with keyword like NumPy
t2D_new = torch.ones(2,3)

# 3-D fully specified
t3D = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

# Finding data type of tensor
print(f'Datatype can be found using name.dtype {t1D.dtype}')

# Shape of tensor found using either 'shape' attribute or 'size()' function
print(f't2D.shape = {t2D.shape} same as t2D.size() = {t2D.size()}')

# Tensor re-shaped with re-shape - may or may not produce another copy
# Below : give all elements to first dim., no second dim. exists in y 
y = t2D.reshape(-1,)
print(f'Re-shaping t2D y = {y.shape}')

# Make z.shape = (6,1) as opposed to (6) above
z = t2D.reshape(-1,1)
print(f'Re-shaping t2D z = {z.shape}')

# Same operation with view
y_v = t2D.view(-1)
print(f'Re-shaping t2D y_v = {y_v.shape}')

# Make z.shape = (6,1) as opposed to (6) above
z_v = t2D.view(-1,1)
print(f'Re-shaping t2D z_v = {z_v.size()}')

# Say now we want to generate a copy of t2D, use new_tensor() function
t2D_new = t2D.new_tensor(t2D)

# PyTorch recommends using clone() + detach() instead of new_tensor()
# detach() removes the tensor from the computation graph (?)
t2D_copy = t2D.clone().detach()

# To convert NumPy array to tensor, preserving dtype
a = np.array([1.0,2.0,3.0])

# Using function as_tensor() shares same data as numpy array
# but because we change the dtype, it makes a copy
t_a = torch.as_tensor(a,dtype=torch.float32)

# Simpler than 'as_tensor()' as it has no dtype value
# Always shares the same memory location as numpy array
t_b = torch.from_numpy(a)




