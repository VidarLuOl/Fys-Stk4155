import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from Functions import OLS, bootstrapOLS, ridgeRegression, lassoRegression, bootstrapRidge, bootstrapLasso

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
terrain2 = imread('SRTM_data_Norway_2.tif')


def dataTerrain(img):
    n = 50
    z = img[:n,:n]      #this is the z
    n = len(z)
    x = np.linspace(0,1,len(z))
    y = np.linspace(0,1,len(z))
    x,y = np.meshgrid(x,y)
    return x,y,z




"""_______OLS_______"""


def plotOLS(x,y,z):
    order = 10
    polynomials = np.linspace(1,order,order)
    
    mse_train = []
    mse_test = []
    r2_train = []
    r2_test = []
    
    
    for p in range(1,order+1):
        data = OLS(x,y,z,p)
        mse_train.append(data[0])
        mse_test.append(data[1])
        r2_train.append(data[2])
        r2_test.append(data[3])
        
    
    plt.figure()
    plt.title("terrain OLS")
    plt.semilogy(polynomials,mse_train, "o", label = "mse train", color = "r")
    plt.semilogy(polynomials,mse_train, color = "r")
    plt.semilogy(polynomials,mse_test, "o", label = "mse test", color = "b")
    plt.semilogy(polynomials,mse_test, color = "b")
    plt.xlabel("order")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
    

"""_______OLS bootstrap_______"""

def plotOLSBootrstrap(x,y,z):
    n_bootstraps = 250
    maxdegree = 16
        
    err,bi,var,polydeg = bootstrapOLS(x,y,z, maxdegree, n_bootstraps)
    
    plt.figure()
    plt.title("OLS bootstrap")
    plt.semilogy(polydeg, err, label='Error', color = "r")
    plt.semilogy(polydeg, bi, label='Bias', color = "g")
    plt.semilogy(polydeg, var, label='Variance', color = "b")
    plt.semilogy(polydeg, err, "o", color = "r")
    plt.semilogy(polydeg, bi, "o", color = "g")
    plt.semilogy(polydeg, var, "o", color = "b")
    plt.xlabel("order")
    plt.ylabel("score")
    plt.legend()
    plt.show()


"""_______Ridge regression______"""
def plotRidge(x,y,z):
    nlambdas = 50
    lam = 0.5
    lambdas = np.logspace(-lam, lam, nlambdas)
    order = 7
    
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
    plt.xlabel("lambda")
    plt.ylabel("score")
    plt.legend()
    plt.show()


def plotRidgeBootstrap(x,y,z):
    n_bootstraps = 1000
    
    nlambdas = 13
    lam_high = 0.5
    lam_low = -12
    order = 8

    err,bi,var, xaxis = bootstrapRidge(x,y,z, order, n_bootstraps, lam_low, lam_high, nlambdas)
    
    plt.figure()
    plt.title("Ridge Bootstrap")
    plt.semilogy(xaxis, err, label='Error', color = "r")
    plt.semilogy(xaxis, bi, label='bias', color = "g")
    plt.semilogy(xaxis, var, label='Variance', color = "b")
    plt.semilogy(xaxis, err, "o", color = "r")
    plt.semilogy(xaxis, bi, "o", color = "g")
    plt.semilogy(xaxis, var, "o", color = "b")
    plt.xlabel("lambda")
    plt.ylabel("score")
    plt.legend()
    plt.show()



""""____Lasso regression______"""

def plotLasso(x,y,z):
    nlambdas = 14
    lam_high = 1
    lam_low = -12
    lambdas = np.logspace(lam_low, lam_high, nlambdas)
    order = 15
        
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
    plt.plot(np.log10(lambdas),error, color = "r")
    plt.plot(np.log10(lambdas),bias, "o", label="bias", color = "g")
    plt.plot(np.log10(lambdas),bias, color = "g")
    plt.plot(np.log10(lambdas),variance, "o", label="variance", color = "b")
    plt.plot(np.log10(lambdas),variance, color = "b")
    plt.xlabel("lambda")
    plt.ylabel("score")
    plt.legend()
    plt.show()


def plotLassoBootstrap(x,y,z):
    n_bootstraps = 500
    
    nlambdas = 27
    lam_high = 1
    lam_low = -12
    order = 5
    err,bi,var, xaxis = bootstrapLasso(x,y,z, order, n_bootstraps, lam_low, lam_high, nlambdas)
    
    plt.figure()
    plt.title("Lasso Bootstrap")
    plt.semilogy(xaxis, err, label='Error', color = "r")
    plt.semilogy(xaxis, bi, label='bias', color = "g")
    #plt.semilogy(xaxis, var, label='Variance', color = "b")
    plt.semilogy(xaxis, err, "o", color = "r")
    plt.semilogy(xaxis, bi, "o", color = "g")
    #plt.semilogy(xaxis, var, "o", color = "b")
    plt.xlabel("lambda")
    plt.ylabel("score")
    plt.legend()
    plt.show()
    
    
    
    
"""____________________________terrain1__________________________"""

x,y,z = dataTerrain(terrain1)  


#plotOLS(x, y, z)
#plotOLSBootrstrap(x, y, z)

plotRidge(x, y, z)
plotRidgeBootstrap(x, y, z)
plotLasso(x, y, z)
#plotLassoBootstrap(x, y, z)

"""______________________________terrain2______________________________"""

x,y,z = dataTerrain(terrain2)

"""
plotOLS(x, y, z)
plotOLSBootrstrap(x, y, z)
plotRidge(x, y, z)
plotRidgeBootstrap(x, y, z)
plotLasso(x, y, z)
"""
