import numpy as np
from Functions import FrankeFunction, OLS, bootstrap, create_X, beta#, cross_validation
import matplotlib.pyplot as plt
import random as random
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def datapoints():
    np.random.seed(20)
    N = 25
    dt = float(1/N)
    x = np.arange(0, 1, dt)
    y = np.arange(0, 1, dt)
    x, y = np.meshgrid(x,y)#
    z_noise = FrankeFunction(x,y) + 0.2*np.random.randn(len(x),len(x))
    return x,y,z_noise


def OLSkFold(maxdegree, kfold):
    x,y,z = datapoints()
    
    mse_train = np.empty((kfold, maxdegree))
    mse_test = np.empty((kfold, maxdegree))
    r2_train = np.empty((kfold, maxdegree))
    r2_test = np.empty((kfold, maxdegree))    
    
    for degree in range(1,maxdegree):
        X = create_X(x,y,degree)
        z_data = np.ravel(z)
        X_train, z_train = shuffle(X, z_data)
        
        
        x_data = np.array_split(X_train, kfolds)
        z_data = np.array_split(z_train, kfolds)
        
    
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
            
            coefs = beta(X_train, z_train)
            
            z_fit = X_train.dot(coefs)
            z_pred = X_test.dot(coefs)
            
            
            MSE_train = mean_squared_error(z_fit, z_train)
            MSE_test = mean_squared_error(z_pred, z_test)
            
            R2_train = r2_score(z_fit, z_train)
            R2_test = r2_score(z_pred, z_test)
            
            mse_train[i][degree] = MSE_train
            mse_test[i][degree] = MSE_test
            r2_train[i][degree] = R2_train
            r2_test[i][degree] = R2_test 

            
    return mse_train, mse_test, r2_train, r2_test

degree = 5
kfolds = 4


mse_train = []
mse_test = []
r2_train = []
r2_test = []

data = OLSkFold(degree, kfolds)

print(data[0][0])

polynoms = np.linspace(1,degree, degree)
       
plt.figure()          
plt.title("mse train")      
for p in range(degree-1):
    plt.plot(polynoms,data[0][p])
    

plt.figure()
plt.title("mse test")                  
for p in range(degree-1):
    plt.plot(polynoms,data[1][p])
    
plt.figure()   
plt.title("r2 train")                
for p in range(degree-1):
    plt.plot(polynoms,data[2][p])

plt.figure() 
plt.title("r2 test")                
for p in range(degree-1):
    plt.plot(polynoms,data[3][p])

    

    

"""
def cross_validation(x,y,z, maxdegree, kfolds):
    error = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)
    
    for degree in range(maxdegree):
        clf = LinearRegression(fit_intercept=False)
        data = OLS(x,y,z,degree)
        
        X_train, X_test, z_train, z_test = data[-4], data[-3], data[-2], data[-1]
        
        z_pred = np.empty((z_test.shape[0], n_bootstraps))
        Z_test = z_pred.copy()
        Z_train = z_train.copy()
        
        
        
        for i in range(n_bootstraps):
            x_, z_ = resample(X_train, Z_train)   
            z_pred[:, i] = clf.fit(x_, z_).predict(X_test)
            Z_test[:, i] = z_test  
            
        
        polydegree[degree] = degree
    
        error[degree] = np.mean( np.mean((Z_test - z_pred)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (Z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
        

        print('Polynomial degree:', degree)
        print('Error:', error[degree])
        print('Bias^2:', bias[degree])
        print('Var:', variance[degree])
        print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

    return error, bias, variance, polydegree
"""