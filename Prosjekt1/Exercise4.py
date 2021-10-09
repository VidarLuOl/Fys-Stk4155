import numpy as np
from Functions import datapoints, create_X, ridge, bootstrapRidge
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



x,y,z = datapoints()


def ridgeRegression(x,y,z,order,lamb):
    X = create_X(x, y, order)
    z_ravel = np.ravel(z)

    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.20)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)

    z_mean_train = np.mean(z_train)
    z_train = (z_train - z_mean_train)/np.std(z_train)
    
    z_mean_test = np.mean(z_test)
    z_train = (z_train - z_mean_test)/np.std(z_test)
    
    coefs = ridge(X_train_scale, z_train, lamb)
    
    z_fit = X_train_scale.dot(coefs)
    z_pred = X_test_scale.dot(coefs)
    
    
    MSE_train_scale = mean_squared_error(z_fit, z_train)
    MSE_test_scale = mean_squared_error(z_pred, z_test)
    
    R2_train_scale = r2_score(z_fit, z_train)
    R2_test_scale = r2_score(z_pred, z_test)
    
    
    error = np.mean(np.mean((z_test - z_pred)**2))
    bias = np.mean((z_test - np.mean(z_pred)))
    variance = np.mean(np.var(z_pred))
    
    data =  MSE_train_scale, MSE_test_scale, \
            R2_train_scale, R2_test_scale, \
            coefs, X_train_scale, X_test_scale, z_train, z_test
            
    return data, error, bias, variance


mse_train = []
mse_test = []
r2_train = []
r2_test = []
error = []
bias = []
variance = []

nlambdas = 50
lam = 0.5
lambdas = np.logspace(-lam, lam, nlambdas)
order = 8
polynomials = np.linspace(1,order,order)

for _ in range(nlambdas):
    data, err, bi, var = ridgeRegression(x,y,z,order,lambdas[_])
    mse_train.append(data[0])
    mse_test.append(data[1])
    r2_train.append(data[2])
    r2_test.append(data[3])
    error.append(err)
    bias.append(bi)
    variance.append(var)


plt.figure()
plt.title("ridge regression")
plt.semilogy(lambdas,error, "o", label="error", color = "r")
plt.semilogy(lambdas,error,color = "r")
plt.semilogy(lambdas,bias, "o", label="bias", color = "g")
plt.semilogy(lambdas,bias, color = "g")
plt.semilogy(lambdas,variance, "o", label="variance", color = "b")
plt.semilogy(lambdas,variance, color = "b")
plt.legend()
plt.show()


n_bootstraps = 150

nlambdas = 13
lam_high = 0.5
lam_low = -12
order = 15

p = [7, 8, 9, 14]
"""
for i in p:
    err,bi,var, xaxis = bootstrapRidge(x,y,z, i, n_bootstraps, lam_low, lam_high, nlambdas)
    
    plt.figure()
    plt.title(i)
    plt.semilogy(xaxis, err, label='Error')
    plt.semilogy(xaxis, bi, label='bias')
    plt.semilogy(xaxis, var, label='Variance')
    plt.legend()
    plt.show()
"""
