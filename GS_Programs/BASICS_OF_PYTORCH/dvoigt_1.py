import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm



#====================================================
# We generate synthetic training data for linear regression model
# y = true_b + true_w * x + epsilon
#====================================================


true_b = 1
true_w = 2
N = 10

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

print(f'original b = {b}, original w = {w}')

#==============================================================================
# Forward Pass: compute the predicted values, we DONT KNOW the value of epsilon
#==============================================================================

yhat = b + w * x_train 

#====================================================
# Compute the Loss i.e. MSE
#====================================================

error = yhat - y_train 
loss = (error ** 2).mean()

print(f'error = {error}')
print(f'loss = {loss}')

#====================================
# Reminder: true_b = 1, true_w = 2
# We want to create a loss surface
# so take a range of b and w values
#====================================

b_range = np.linspace(true_b - 3, true_b + 3, 101) 
w_range = np.linspace(true_w - 3, true_w + 3, 101)
bs, ws = np.meshgrid(b_range, w_range)

#========================================
# Find prediction for a single data-point
# using each combination 'b' and 'w'
#========================================

dummy_x = x_train[0]
dummy_yhat = bs + dummy_x * ws # A = B + alpha * C

print(f'x_train shape = {x_train.shape}')

#========================================
# Predictions for all training points
# i.e. x[i] * ws + bs
#========================================
all_predictions = np.apply_along_axis(func1d=lambda x: bs + x * ws, axis=1,arr=x_train)

#====================================================
# Every 2-D matrix in all_predictions contains
# predictions at every (b,w) combination
# Need to subtract true label from every (b,w) point 
# but y_train needs to be made compatible
#====================================================

all_labels = y_train.reshape(-1,1,1) # Now 3-D matrix
all_errors = (all_predictions - all_labels)
all_losses = (all_errors **2).mean(axis=0)

#=====================================================
# Plotting the loss surface i.e. values stored in 
# all_losses (z-coordinate) for every combination
# of b, w values in bs, ws
#=====================================================

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
surf = ax.plot_surface(bs, ws, all_losses, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# plt.show() <--- to see plot uncomment

#======================================================
# Now switch back to the regular error, loss surface was 
# for education purposes only ;) - compute gradients
#=======================================================
b_grad = 2 * error.mean()
w_grad = 2 * (error * x_train).mean() 

#=======================================================
# Update 'b' and 'w' parameters
#=======================================================
print(f'original b = {b}, original w = {w}')
lr = 0.1
b = b - lr * b_grad
w = w - lr * w_grad

print(f'updated b = {b}, updated w = {w}')
