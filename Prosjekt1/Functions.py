#Functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



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



def beta(X,y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def OLS(x,y,z,order):
    MSE_train = []
    R2_train = []
    MSE_test = []
    R2_test = []
    coefs = []

    MSE_train_scale = []
    R2_train_scale = []
    MSE_test_scale = []
    R2_test_scale = []
    coefs_scale = []
    
    X_train_scaled = []
    
    for p in order:
        X = create_X(x, y, p)
        z_ravel = np.ravel(z)     
        X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.20)
        
        poly = PolynomialFeatures(degree=p)
        X = poly.fit_transform(z)
        clf = LinearRegression(fit_intercept=False)
        
        clf = LinearRegression().fit(X_train, z_train)
        
        z_fit = clf.predict(X_train)
        z_pred = clf.predict(X_test)
        
        coefs.append((clf.coef_))

        
        MSE_train.append(mean_squared_error(z_fit, z_train))
        R2_train.append(r2_score(z_fit, z_train))
        MSE_test.append(mean_squared_error(z_pred, z_test))
        R2_test.append(r2_score(z_pred, z_test))
        
        
        scaler = StandardScaler(with_std=False)
        scaler.fit(X_train)
    
        X_train_scale = scaler.transform(X_train)
        X_test_scale = scaler.transform(X_test)
        
        X_train_scaled.append(X_train_scale)
        
        clf.fit(X_train_scale, z_train)
        
        z_fit_scale = clf.predict(X_train_scale)
        z_pred_scale = clf.predict(X_test_scale)
        
        coefs_scale.append((clf.coef_))
        
        MSE_train_scale.append(mean_squared_error(z_fit_scale, z_train))
        R2_train_scale.append(r2_score(z_fit_scale, z_train))
        MSE_test_scale.append(mean_squared_error(z_pred_scale, z_test))
        R2_test_scale.append(r2_score(z_pred_scale, z_test))
        
    

    
    data = list((MSE_train, R2_train, MSE_test, R2_test, coefs, MSE_train_scale, \
                 R2_train_scale, MSE_test_scale, R2_test_scale, coefs_scale, X_train_scaled))
    return data


def plot(x,y,label,ylab):
    plt.figure()
    plt.title(label)
    plt.xlabel("polynom order")
    plt.ylabel(ylab)
    plt.plot(x,y)
    plt.show()
    
    
    
