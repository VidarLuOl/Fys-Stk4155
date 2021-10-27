# Importing various packages
from functions import datapoints, create_X, OLS, SGD
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import random as random



n_epochs = 50
M = 5
eta = 0.0001
t0, t1 = 0, 1

mse_train_SGD = []
mse_test_SGD = []
mse_train_OLS = []
mse_test_OLS = []



for n in range(12):
    np.random.seed(1997)
    x,y,z = datapoints(150)
    X = create_X(x, y, n)
    X_train, X_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size = 0.20)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)
    
    theta = SGD(X_train_scale, z_train, n_epochs, M, eta, t0, t1)
    
    z_fit_SGD = X_train_scale@theta
    z_pred_SGD = X_test@theta
    
    mse_train_SGD.append(mean_squared_error(z_fit_SGD, z_train))
    mse_test_SGD.append(mean_squared_error(z_pred_SGD, z_test))

    beta = OLS(X_train_scale, z_train)
    
    z_fit_OLS = X_train_scale@beta
    z_pred_OLS = X_test@beta
    
    mse_train_OLS.append(mean_squared_error(z_fit_OLS, z_train))
    mse_test_OLS.append(mean_squared_error(z_pred_OLS, z_test))

p = np.linspace(0, n+1, n+1)


plt.figure()
plt.plot(p, mse_train_SGD, label= "SGD", color = "r")
plt.scatter(p, mse_train_SGD, color = "r")
plt.plot(p, mse_train_OLS, label= "OLS", color = "b")
plt.scatter(p, mse_train_OLS, color = "b")
plt.title("MSE training")
plt.xlabel("order")
plt.ylabel("score")
plt.legend()
plt.show()


