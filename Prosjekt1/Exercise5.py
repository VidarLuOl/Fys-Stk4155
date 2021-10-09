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
    
    RegLasso = linear_model.Lasso(lamb,fit_intercept=False)
    RegLasso.fit(X_train_scale,z_train)
    
    #z_tilde = RegLasso.predict(X_train)
    z_pred = RegLasso.predict(X_test)
    coefs = RegLasso.coef_


    
    error = np.mean(np.mean((z_test - z_pred)**2))
    bias = np.mean((z_test - np.mean(z_pred)))
    variance = np.mean(np.var(z_pred))
            
    return error, bias, variance, coefs




nlambdas = 13
lam_high = 0.5
lam_low = -12
lambdas = np.logspace(lam_low, lam_high, nlambdas)
order = 10
polynomials = np.linspace(1,order,order)


error = []
bias = []
variance = []
coefs = []

for p in lambdas:
    data = lassoRegression(x,y,z,order,p, nlambdas)
    i,j,k,l  = data[0], data[1], data[2], data[3]
    error.append(i)
    bias.append(j)
    variance.append(k)
    coefs.append(l)

"""
plt.figure()
plt.title("lasso regression")
plt.plot(np.log10(lambdas),error, label="error")
plt.plot(np.log10(lambdas),bias, label="bias")
plt.plot(np.log10(lambdas),variance, label="variance")
plt.show()
"""

"""
plt.figure()
plt.title("lasso regression")
plt.plot(lambdas,error, label="error")
plt.plot(lambdas,bias, label="bias")
plt.plot(lambdas,variance, label="variance")
plt.show()
"""
