#Functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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

def OLS(x,y,z,order):
    X = create_X(x, y, order)
    z_ravel = np.ravel(z)

    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.20)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)

    X_train_scale[:,0] = 1
    X_test_scale[:,0] = 1
    
    
    coefs = beta_ols(X_train_scale, z_train)
    
    z_fit = X_train_scale.dot(coefs)
    z_pred = X_test_scale.dot(coefs)
    
    
    MSE_train_scale = mean_squared_error(z_fit, z_train)
    MSE_test_scale = mean_squared_error(z_pred, z_test)
    
    R2_train_scale = r2_score(z_fit, z_train)
    R2_test_scale = r2_score(z_pred, z_test)
    
    
    data =  MSE_train_scale, MSE_test_scale, \
            R2_train_scale, R2_test_scale, \
            coefs, z_fit, z_pred
            
    return data


def SGD(X_data, z_data, n_epochs, batch_size, eta):
    a,b = X_data.shape
    m = len(z_data)
    
    theta = np.random.randn(b)
    
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(a-5)
            xi = X_data[random_index:random_index+5]
            zi = z_data[random_index:random_index+5]
            gradients = 2.0* xi.T @ ((xi @ theta)-zi)
            theta = theta - eta*gradients
            
    return theta


def sigmoid(x):
    return 1/(1 + np.exp**(-x))