# Importing various packages
from functions import datapoints, create_X, sigmoid
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random as random

np.random.seed(2021)

n = 9
x,y,z = datapoints(N=50)
X = create_X(x, y, n)


z_ravel = np.ravel(z)

X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size = 0.20)
    

a, b = X_train.shape

n_epochs = 50
M = 5   #size of each minibatch
m = len(z_train)
eta = 0.0001

theta = np.random.randn(b)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(a-5)
        xi = X_train[random_index:random_index+5]
        zi = z_train[random_index:random_index+5]
        gradients = 2.0* xi.T @ ((xi @ theta)-zi)
        theta = theta - eta*gradients



ypredict3 = X_test.dot(theta)



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.scatter(X_test[:,1], X_test[:,2], ypredict3)


np.random.seed(2021)

def SGD(X_data, z_data, n_epochs, M, eta):
    a,b = X_train.shape
    m = len(z_data)
    index = np.arange(0,m)
    theta = np.random.randn(b)
    for epoch in range( n_epochs):
        random.shuffle(index)
        X_train_sh = X_data[index]
        z_train_sh = z_data[index]
        
        for i in range(0,m,M):
            xi = X_train_sh[i:i+M]
            zi = z_train_sh[i:i+M]
            gradients = 2.0*xi.T @ ((xi @ theta)-zi)
            theta = theta - eta*gradients

    
    return theta

theta = SGD(X_train, z_train, n_epochs, M, eta)

ypred = X_test.dot(theta)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.scatter(X_test[:,1], X_test[:,2], ypred)

