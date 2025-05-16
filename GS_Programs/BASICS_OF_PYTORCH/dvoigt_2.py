#==========================================================================
# This program is same as dvoigt_1.py
# except I am removing plotting the loss surface for a range of 'b' and 'w'. 
# also specifying epochs and looping through them
#==========================================================================

import numpy as np
#====================================================
# We generate synthetic training data for linear regression model
# y = true_b + true_w * x + epsilon
#====================================================

true_b = 1
true_w = 2
N = 100

# Fix the seed to generate same sequence of pseudo-random numbers
np.random.seed(42)

# Generate N x 1 column vector 'x'
x = np.random.rand(N,1)

# Generate 'noise' but make it small i.e. multiply by 0.1
epsilon = 0.1 * np.random.randn(N,1)

y = true_b + true_w * x + epsilon

#====================================================
# Divide samples into train-test sets
#====================================================

# Generate [0,N) numbers in a NumPy array
idx = np.arange(N)
np.random.shuffle(idx)

train_ids = idx[:int(N * 0.8)]
test_ids  = idx[int(0.8 * N) :]

x_train, y_train = x[train_ids], y[train_ids]
x_test, y_test   = x[test_ids],  y[test_ids]

#====================================================
# Randomly initialize 'w' and 'b' 
#====================================================

w = np.random.randn(1)
b = np.random.randn(1)

print(f'initial random b = {b}, initial random w = {w}')

#==============================================================================
# Forward Pass: compute the predicted values, we DONT KNOW the value of epsilon
#==============================================================================

n_epochs = 500
lr = 0.1


for epoch in range(n_epochs):
    # First predict using forward pass
    yhat = b + w * x_train 

    # Compute the Loss i.e. MSE
    error = yhat - y_train 
    loss = (error ** 2).mean()

    # Compute Gradients
    b_grad = 2 * error.mean()
    w_grad = 2 * (error * x_train).mean() 

    # Update 'b' and 'w' parameters
    b = b - lr * b_grad
    w = w - lr * w_grad

    # print the predicted b,w and calculated loss
    if epoch % 25 == 0:
        print(f'updated b = {b}, updated w = {w}, loss = {loss}')

print(f'True b = {true_b}, True w = {true_w}')
