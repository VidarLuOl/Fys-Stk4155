import numpy as np
from Functions import datapoints, create_X, beta
from Funksjoner import Variance

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



x,y,z = datapoints()

order = 15
polynomials = np.linspace(1,order,order)

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
    data = OLS(x,y,z,p)
    i,j,k,l = data[0], data[1], data[2], data[3]
    mse_train.append(i)
    mse_test.append(j)
    r2_train.append(k)
    r2_test.append(l)
    print("Polynom: %.f | MSE train: %.3f | MSE test: %.3f | R2 train: %.3f | R2 test: %.3f" %(p,i,j,k,l))

"""
plt.figure()
plt.title("mse")
plt.semilogy(polynomials,mse_train)
plt.semilogy(polynomials,mse_test)
plt.show()

plt.figure()
plt.title("r2")
plt.plot(polynomials,r2_train)
plt.plot(polynomials,r2_test)        
plt.show()
"""


plt.figure()
plt.title("mse OLS")
plt.semilogy(polynomials,mse_train, "o", label = "mse train", color = "r")
plt.semilogy(polynomials,mse_train, color = "r")
plt.semilogy(polynomials,mse_test, "o", label = "mse test", color = "b")
plt.semilogy(polynomials,mse_test, color = "b")
plt.xlabel("order")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.figure()
plt.title("R2 OLS")
plt.plot(polynomials,r2_train, "o", label = "R2 train", color = "r")
plt.plot(polynomials,r2_train, color = "r")
plt.plot(polynomials,r2_test, "o", label = "R2 test", color = "b")
plt.plot(polynomials,r2_test, color = "b")
plt.xlabel("order")
plt.ylabel("R2")
plt.legend()
plt.show()
