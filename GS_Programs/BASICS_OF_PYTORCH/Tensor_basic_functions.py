import torch

t2D = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]) # 2 x 4 matrix
print(f'Shape = {t2D.shape}') # 2 x 4 

print(f'Re-shaped tensor using "reshape" \n = {t2D.reshape(4,2)}') 
# Output =  [[1,2], [3,4], [5,6], [7,8]] 

print(f'Re-shaped tensor using "view" \n = {t2D.view(4,2)}')
# Output =  [[1,2], [3,4], [5,6], [7,8]]

print(f'Transpose of Tensor \n = {t2D.T}')
# Output = [ [1,5], [2,6], [3,7], [4,8] ]

print(f'Matrix Multiplication = {t2D @ t2D.T}') # First syntax
# Output = [[ 30,  70],[ 70, 174]])

print(f'Matrix Multiplication = {t2D.matmul(t2D.T)}') # Second syntax
# Output = [[ 30,  70],[ 70, 174]]
