import numpy as np
from Functions import datapoints, create_X, ridge, bootstrapRidge
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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

def RidgeRegressionkFold(x,y,z, order, kfold, lam):   
    mse_train = np.empty((order, kfold))
    mse_test = np.empty((order, kfold))
    r2_train = np.empty((order, kfold))
    r2_test = np.empty((order, kfold))         
    

    X = create_X(x,y,order)
    z_data = np.ravel(z)
    X_train, z_train = shuffle(X, z_data)
        
        
    x_data = np.array_split(X_train, kfold)
    z_data = np.array_split(z_train, kfold)
        
    
    for i in range(0,len(x_data)):    
        X_test = x_data[i]
        X_train = x_data.copy()
        X_train.pop(i)
        X_train = np.concatenate(X_train, axis=0)
            
        z_test = z_data[i]
        z_train = z_data.copy()
        z_train.pop(i)
        z_train = np.concatenate(z_train, axis=0)
            
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
            
        coefs = ridge(X_train_scale, z_train, lam)
            
        z_fit = X_train_scale.dot(coefs)
        z_pred = X_test_scale.dot(coefs)
            
            
        MSE_train = mean_squared_error(z_fit, z_train)
        MSE_test = mean_squared_error(z_pred, z_test)
        
        R2_train = r2_score(z_fit, z_train)
        R2_test = r2_score(z_pred, z_test)

        mse_train[i] = MSE_train
        mse_test[i] = MSE_test
        r2_train[i] = R2_train
        r2_test[i] = R2_test    
            
    return mse_train, mse_test, r2_train, r2_test

"""
mse_train = []
mse_test = []
r2_train = []
r2_test = []
error = []
bias = []
variance = []

nlambdas = 25
lambdas = np.logspace(-12, 0.5, nlambdas)
order = 7
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
plt.title("Ridge regression")
plt.semilogy(lambdas,error, "o", label="error", color = "r")
plt.semilogy(lambdas,error,color = "r")
plt.semilogy(lambdas,bias, "o", label="bias", color = "g")
plt.semilogy(lambdas,bias, color = "g")
plt.semilogy(lambdas,variance, "o", label="variance", color = "b")
plt.semilogy(lambdas,variance, color = "b")
plt.legend()
plt.show()
"""

n_bootstraps = 625


nlambdas = 15
lam_high = 0.5
lam_low = -12
order = 5

#p = [7, 8, 9, 14]

err,bi,var, xaxis = bootstrapRidge(x,y,z, order, n_bootstraps, lam_low, lam_high, nlambdas)
    
plt.figure()
plt.title("Ridge regression bootstrap")
plt.semilogy(xaxis, err, label='error', color="r")
plt.semilogy(xaxis, bi, label='bias', color="g")
plt.semilogy(xaxis, var, label='variance', color="b")
plt.semilogy(xaxis, err, "o", color="r")
plt.semilogy(xaxis, bi, "o", color="g")
plt.semilogy(xaxis, var, "o", color="b")
plt.legend()
plt.show()
"""


kfolds = 5

nlambdas = 25
lambdas = np.logspace(-12, 0.5, nlambdas)
order = 7
polynomials = np.linspace(1,order,order)

for _ in range(nlambdas):
    msetrain, msetest, r2train, r2test = RidgeRegressionkFold(x,y,z,order, kfolds, lambdas[_])
    msetrain = [np.mean(msetrain[_]) for _ in range(order)]
    msetest = [np.mean(msetest[_]) for _ in range(order)]
    r2train = [np.mean(r2train[_]) for _ in range(order)]
    r2test = [np.mean(r2test[_]) for _ in range(order)]

polynoms = np.linspace(1,order,order)

plt.figure()         
plt.title("MSE kFold Ridge") 
plt.plot(polynoms,msetrain, label = "MSE train", color = "r")                
plt.plot(polynoms,msetest, label = "MSE test", color = "b")
plt.plot(polynoms,msetrain, "o", color = "r")                
plt.plot(polynoms,msetest, "o", color = "b")  
plt.xlabel("order")
plt.ylabel("mse")
plt.legend()
plt.show()  
    
plt.figure()         
plt.title("R2 kFold Ridge") 
plt.plot(polynoms,r2train, label = "R2 train", color = "r")                
plt.plot(polynoms,r2test, label = "R2 test", color = "b")
plt.plot(polynoms,r2train, "o", color = "r")                
plt.plot(polynoms,r2test, "o", color = "b")  
plt.xlabel("order")
plt.ylabel("R2")
plt.legend()
plt.show() 
"""