import numpy as np
from Functions import FrankeFunction, create_X, beta#, OLS
from Funksjoner import Variance
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample



np.random.seed(20)

N = 25
dt = float(1/N)
x = np.arange(0, 1, dt)
y = np.arange(0, 1, dt)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x,y)
z_noise = FrankeFunction(x,y) + 0.2*np.random.randn(len(x),len(x))
order = 8

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

mse_train = []
mse_test = []
r2_train = []
r2_test = []


for p in range(1,order+1):
    data = OLS(x,y,z_noise,p)
    i,j,k,l = data[0], data[1], data[2], data[3]
    mse_train.append(i)
    mse_test.append(j)
    r2_train.append(k)
    r2_test.append(l)
    print("Polynom: %.f | MSE train: %.3f | MSE test: %.3f | R2 train: %.3f | R2 test: %.3f" %(p,i,j,k,l))



polynomials = np.linspace(1,order,order)

    
    
plt.figure()
plt.semilogy(polynomials,mse_train)
plt.semilogy(polynomials,mse_test)


plt.figure()
plt.plot(polynomials,r2_train)
plt.plot(polynomials,r2_test)        



"""
data5 = OLS(x,y,z_noise,5)

beta5 = data5[-5]
X = data5[-4]

var = Variance(X,20)

sigma = np.sqrt(var)*1.96

plt.figure()
x_axis = np.linspace(0,len(beta5),len(beta5))
plt.errorbar(x_axis, beta5, sigma, fmt="o")
"""