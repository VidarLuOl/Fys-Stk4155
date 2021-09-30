#Functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample




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
    X = create_X(x, y, order)
    z_ravel = np.ravel(z)

    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.20)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)

    X_train_scale[:,0] = 1
    X_test_scale[:,0] = 1
    
    
    coefs = beta(X_train_scale, z_train)
    
    z_fit = X_train_scale.dot(coefs)
    z_pred = X_test_scale.dot(coefs)
    
    
    MSE_train_scale = mean_squared_error(z_fit, z_train)
    MSE_test_scale = mean_squared_error(z_pred, z_test)
    
    R2_train_scale = r2_score(z_fit, z_train)
    R2_test_scale = r2_score(z_pred, z_test)
    
    
    data =  MSE_train_scale, MSE_test_scale, \
            R2_train_scale, R2_test_scale, \
            coefs, X_train_scale, X_test_scale, z_train, z_test
            
    return data


def OLS_sklearn(x,y,z,order):
    
    X = create_X(x, y, order)
    z_ravel = np.ravel(z)     
    X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.20)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scale = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)
    
    clf = LinearRegression(fit_intercept=False)    
    clf.fit(X_train_scale, z_train)
        
    z_fit_scale = clf.predict(X_train_scale)
    z_pred_scale = clf.predict(X_test_scale)
    
    coefs_scale = clf.coef_
    
    MSE_train_scale = mean_squared_error(z_fit_scale, z_train)
    MSE_test_scale = mean_squared_error(z_pred_scale, z_test)
    
    R2_train_scale = r2_score(z_fit_scale, z_train)
    R2_test_scale = r2_score(z_pred_scale, z_test)
        
    

    
    #data = coefs, X_train, MSE_train, R2_train, MSE_test, R2_test
    data =  MSE_train_scale, MSE_test_scale, \
                   R2_train_scale, R2_test_scale, \
                   coefs_scale, X_train_scale, X_test_scale, z_train, z_test
            
    return data



    
    
def boot(data, datapoints):
    t = np.zeros(datapoints)
    n = len(data)
    # non-parametric bootstrap         
    for i in range(datapoints):
        t[i] = np.mean(data[np.random.randint(0,n,n)])
    # analysis    
    #print("Bootstrap Statistics :")
    #print("original           bias      std. error")
    #print("%8g %8g %14g %15g" % (np.mean(data), np.std(data),np.mean(t),np.std(t)))
    return t  
    
def bootstrap(x,y,z, maxdegree, n_bootstraps):
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

        
        for i in range(n_bootstraps):
            x_, z_ = resample(X_train, z_train)   
            z_pred[:, i] = clf.fit(x_, z_).predict(X_test)
            Z_test[:, i] = z_test  
            
        
        polydegree[degree] = degree
    
        error[degree] = np.mean( np.mean((Z_test - z_pred)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (Z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
        
        """
        print('Polynomial degree:', degree)
        print('Error:', error[degree])
        print('Bias^2:', bias[degree])
        print('Var:', variance[degree])
        print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
        """
    return error, bias, variance, polydegree