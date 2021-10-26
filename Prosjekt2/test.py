# Importing various packages
from functions import datapoints, create_X, sigmoid
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

n = 7
x,y,z = datapoints()
X = create_X(x, y, n)


z_ravel = np.ravel(z)

X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.20)
    
scaler = StandardScaler()
scaler.fit(X_train)
    
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

X_train_scale[:,0] = 1
X_test_scale[:,0] = 1


scaler = StandardScaler()
X_scale = scaler.fit(X_train)
X_train_scale = scaler.transform(X)
    
theta_linreg = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ z_train)
#print("Own inversion")
#print(theta_linreg)
sgdreg = SGDRegressor(max_iter = 50, penalty=None, eta0=0.1)
sgdreg.fit(X_train,z_train)
#print("sgdreg from scikit")
#print(sgdreg.intercept_, sgdreg.coef_)


a,b = np.shape(X_train_scale)

theta = np.random.randn(b,a)
eta = 0.0001
Niterations = 100
 

for iter in range(Niterations):
    gradients = 2.0/n*X_train_scale.T @ ((X_train_scale @ theta)-z_ravel)
    theta -= eta*gradients
#print("theta from own gd")
#print(theta)


xnew = X_test.copy()
ypredict = xnew.dot(theta)
ypredict2 = xnew.dot(theta_linreg)


n_epochs = 50
M = 5   #size of each minibatch
m = int(n/M) #number of minibatches
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(b)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(a-5)
        xi = X_train[random_index:random_index+5]
        zi = z_train[random_index:random_index+5]
        gradients = 2.0* xi.T @ ((xi @ theta)-zi)
        theta = theta - eta*gradients
#print("theta from own sdg")
#print(theta)

ypredict3 = xnew.dot(theta)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.scatter(xnew[:,1], xnew[:,2], ypredict3)

"""
plt.scatter(xnew, ypredict2, "b-")
plt.scatter(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Random numbers ')
plt.show()
"""