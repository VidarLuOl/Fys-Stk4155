from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from Functions import FrankeFunction, create_X, beta, plot
from Exercise1 import OLS
from sklearn.linear_model import LinearRegression, Ridge, Lasso

np.random.seed(2021)


x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x,y)
z_noise = FrankeFunction(x,y) + 0.05*np.random.randn(20,20)

#P = 15
#polynomials = [_ for _ in range(1,P+1)]
polynomials = [15]
data = OLS(x,y,z_noise,polynomials)





    
plot(polynomials,data[0], "MSE train", "mse score")

#plot(polynomials,data[1], "R2 train", "r2 score")
plot(polynomials,data[2], "MSE test", "mse score")
#plot(polynomials,data[3], "R2 test", "r2 score")

"""
plot(polynomials,data[5], "MSE train scaled", "mse score")
plot(polynomials,data[6], "R2 train scaled", "r2 score")
plot(polynomials,data[7], "MSE test scaled", "mse score")
plot(polynomials,data[8], "R2 test scaled", "r2 score")        
"""