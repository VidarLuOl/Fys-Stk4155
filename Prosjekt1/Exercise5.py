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
    
    z_tilde = RegLasso.predict(X_train)
    z_pred = RegLasso.predict(X_test)
    coefs = RegLasso.coef_

    MSE_train_scale = mean_squared_error(z_tilde, z_train)
    MSE_test_scale = mean_squared_error(z_pred, z_test)
    
    R2_train_scale = r2_score(z_tilde, z_train)
    R2_test_scale = r2_score(z_pred, z_test)

    
    error = np.mean(np.mean((z_test - z_pred)**2))
    bias = np.mean((z_test - np.mean(z_pred)))
    variance = np.mean(np.var(z_pred))
            
    return error, bias, variance, coefs, MSE_train_scale, MSE_test_scale, R2_train_scale, R2_test_scale




nlambdas = 13
lam_high = 0.5
lam_low = -12
lambdas = np.logspace(lam_low, lam_high, nlambdas)
order = 15

polynomials = np.linspace(1,order,order)


error = []
bias = []
variance = []
coefs = []
msetrain = []
msetest = []
r2test = []
r2train = []

for p in lambdas:
    data = lassoRegression(x,y,z,order,p, nlambdas)
    error.append(data[0])
    bias.append(data[1])
    variance.append(data[2])
    coefs.append(data[3])
    msetrain.append(data[4])
    msetest.append(data[5])
    r2train.append(data[6])
    r2test.append(data[7])
    


plt.figure()
plt.title("lasso regression")
plt.plot(np.log10(lambdas),error, "o", label="error", color = "r")
plt.plot(np.log10(lambdas),error,color = "r")
plt.plot(np.log10(lambdas),bias, "o", label="bias", color = "g")
plt.plot(np.log10(lambdas),bias, color = "g")
plt.plot(np.log10(lambdas),variance, "o", label="variance", color = "b")
plt.plot(np.log10(lambdas),variance, color = "b")
plt.legend()
plt.show()


plt.figure()
plt.title("lasso regression mse")
plt.plot(np.log10(lambdas),msetrain, "o", label="mse train", color = "r")
plt.plot(np.log10(lambdas),msetrain,color = "r")
plt.plot(np.log10(lambdas),msetest, "o", label="mse test", color = "b")
plt.plot(np.log10(lambdas),msetest, color = "b")
plt.legend()
plt.show()

plt.figure()
plt.title("lasso regression R2")
plt.plot(np.log10(lambdas),r2train, "o", label="mse train", color = "r")
plt.plot(np.log10(lambdas),r2train,color = "r")
plt.plot(np.log10(lambdas),r2test, "o", label="mse test", color = "b")
plt.plot(np.log10(lambdas),r2test, color = "b")
plt.legend()
plt.show()