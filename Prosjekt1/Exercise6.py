import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from Functions import create_X, OLS, FrankeFunction, bootstrapOLS

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
terrain2 = imread('SRTM_data_Norway_2.tif')
"""
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""

def dataTerrain(img):
    n = 50
    z = img[:n,:n]      #this is the z
    n = len(z)
    x = np.linspace(0,1,len(z))
    y = np.linspace(0,1,len(z))
    x,y = np.meshgrid(x,y)
    return x,y,z



"""____________________________terrain1_________________________"""

order = 11

x,y,z = dataTerrain(terrain1)

polynomials = np.linspace(1,order,order)

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

plt.figure()
plt.title("terrain1 OLS")
plt.semilogy(polynomials,mse_train, "o", label = "mse train", color = "r")
plt.semilogy(polynomials,mse_train, color = "r")
plt.semilogy(polynomials,mse_test, "o", label = "mse test", color = "b")
plt.semilogy(polynomials,mse_test, color = "b")
plt.xlabel("order")
plt.ylabel("MSE")
plt.legend()
plt.show()

n_bootstraps = 250
maxdegree = 15
    
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



"""______________________________terrain2______________________________"""

x,y,z = dataTerrain(terrain2)

polynomials = np.linspace(1,order,order)

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

plt.figure()
plt.title("terrain2 OLS")
plt.semilogy(polynomials,mse_train, "o", label = "mse train", color = "r")
plt.semilogy(polynomials,mse_train, color = "r")
plt.semilogy(polynomials,mse_test, "o", label = "mse test", color = "b")
plt.semilogy(polynomials,mse_test, color = "b")
plt.xlabel("order")
plt.ylabel("MSE")
plt.legend()                                                                    
plt.show()


n_bootstraps = 250
maxdegree = 15
    
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