import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8,12) 
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12,8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8,1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x
    


# print(model)
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
X=dataset[:,0:8]    # print(X.shape) prints (768,8)
Y=dataset[:,8]      # print(Y.shape) prints (768,) <--- vector

# NumPy defaults to FP64 while PyTorch uses FP32 => prefer conversion

X=torch.tensor(X,dtype=torch.float32)

#Torch prefers nx1 matrix instead of n-vector, so convert (768,) --> (768,1)

Y=torch.tensor(Y,dtype=torch.float32).reshape(-1,1) 

model = PimaClassifier()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 10
batch_size = 20

for i in range(epochs):
    for j in range(0,len(X),batch_size):
        Xbatch = X[j:j+batch_size]
        Ytrue  = Y[j:j+batch_size]
        Ypred = model(Xbatch)
        loss = loss_fn(Ypred,Ytrue)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {i}, Loss = {loss}')

# Evaluate the training accuracy by checking if predicted value matches actual label

with torch.no_grad():
    y_pred = model(X)

accuracy = (y_pred.round() == Y).float().mean()
print(f'Model accuracy on training data = {accuracy}')




