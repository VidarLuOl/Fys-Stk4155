#Functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def datapoints(N=25):
    np.random.seed(124)
    dt = float(1/N)
    x = np.arange(0, 1, dt)
    y = np.arange(0, 1, dt)
    x, y = np.meshgrid(x,y)
    z_noise = FrankeFunction(x,y) + 0.2*np.random.randn(len(x),len(x))
    return x,y,z_noise

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X



def beta_ols(X,y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

def beta_ridge(X,y,lamb):
    I = np.eye(len(X[0]),len(X[0]))
    return np.linalg.inv(X.T.dot(X)-lamb*I).dot(X.T).dot(y)

def OLS(X_data, z_data):  
    clf = LinearRegression(fit_intercept=False)    
    clf.fit(X_data, z_data)
    coefs = clf.coef_
            
    return coefs

def learning_schedule(t0, t1, t):
    return t0/(t+t1)

def SGD(X_data, z_data, n_epochs, M, eta, t0, t1):
    a,b = X_data.shape
    m = len(z_data)
    print(m)
    index = np.arange(0,m)
    theta = np.random.randn(b)
    for epoch in range(n_epochs):
        np.random.shuffle(index)
        X_train_sh = X_data[index]
        z_train_sh = z_data[index]
        
        for i in range(0,m,M):
            xi = X_train_sh[i:i+M]
            zi = z_train_sh[i:i+M]
            gradients = 2.0*xi.T @ ((xi @ theta)-zi)
            if t0 != 0:
                eta = learning_schedule(t0, t1, epoch*m+i)
            
            theta = theta - eta*gradients

    
    return theta


def sigmoid(x):
    return 1/(1 + np.exp**(-x))