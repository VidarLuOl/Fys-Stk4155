import numpy as np
from Functions import datapoints, create_X
from Funksjoner import Variance

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model

x,y,z = datapoints()

def lassoRegression(x,y,z, order, lamb, nlambdas):
    error = np.zeros(nlambdas)
    bias = np.zeros(nlambdas)
    variance = np.zeros(nlambdas)
    
    X = create_X(x, y, order)
    z_ravel = np.ravel(z)

    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.20)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)
    
    X_train_scale[:,0] = 1
    X_test_scale[:,0] = 1
    
    z_mean_train = np.mean(z_train)
    z_train = (z_train - z_mean_train)/np.std(z_train)
    
    z_mean_test = np.mean(z_test)
    z_train = (z_train - z_mean_test)/np.std(z_test)
    
    RegLasso = linear_model.Lasso(lamb)
    RegLasso.fit(X_train_scale,z_train)
    
    z_tilde = RegLasso.predict(X_train)
    z_pred = RegLasso.predict(X_test)
    #coefs = RegLasso.coef_

    
    MSE_train_scale = mean_squared_error(z_tilde, z_train)
    MSE_test_scale = mean_squared_error(z_pred, z_test)
    
    R2_train_scale = r2_score(z_tilde, z_train)
    R2_test_scale = r2_score(z_pred, z_test)
    
    
    print(np.shape(z_pred), np.shape(z_test))
    
    error[nlambdas] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias[nlambdas] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[nlambdas] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
            
    return error, bias, variance



mse_train = []
mse_test = []
r2_train = []
r2_test = []


nlambdas = 13
lam_high = 0.5
lam_low = -12
lambdas = np.logspace(lam_low, lam_high, nlambdas)
order = 15
polynomials = np.linspace(1,order,order)


mse_train = []
mse_test = []
r2_train = []
r2_test = []


for p in lambdas:
    data = lassoRegression(x,y,z,order,p, nlambdas)
    i,j,k,l = data[0], data[1], data[2], data[3]
    mse_train.append(i)
    mse_test.append(j)
    r2_train.append(k)
    r2_test.append(l)


plt.figure()
plt.title("mse")
plt.plot(np.log10(lambdas),mse_train)
plt.plot(np.log10(lambdas),mse_test)
plt.show()

"""
plt.figure()
plt.title("r2")
plt.plot(np.log10(lambdas),r2_train)
plt.plot(np.log10(lambdas),r2_test)    
plt.show()
"""