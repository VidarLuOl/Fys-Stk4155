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
from sklearn.linear_model import LinearRegression, Ridge, Lasso

np.random.seed(2021)


x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x,y)
z_noise = FrankeFunction(x,y) + 0.05*np.random.randn(20,20)


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
        
    
        clf.fit(X_train_scale, z_train)
        
        z_fit_scale = clf.predict(X_train_scale)
        z_pred_scale = clf.predict(X_test_scale)
        
        coefs.append((clf.coef_))
        
        MSE_train_scale.append(mean_squared_error(z_fit_scale, z_train))
        R2_train_scale.append(r2_score(z_fit_scale, z_train))
        MSE_test_scale.append(mean_squared_error(z_pred_scale, z_test))
        R2_test_scale.append(r2_score(z_pred_scale, z_test))
        
    

    
    data = list((MSE_train, R2_train, MSE_test, R2_test, coefs, MSE_train_scale, \
                 R2_train_scale, MSE_test_scale, R2_test_scale, coefs_scale))
    return data


P = 10
polynomials = [_ for _ in range(1,P+1)]

data = OLS(x,y,z_noise,polynomials)

print("Normal data")
for p,i,j,k,l in zip(polynomials,data[0], data[2], data[1], data[3]):
    print("Polynom: %.f | MSE train: %.3f | MSE test: %.3f | R2 train: %.3f | R2 test: %.3f)" %(p,i,j,k,l))
    
    
print("Scaled data")
for p,i,j,k,l in zip(polynomials,data[5], data[7], data[6], data[8]):
    print("Polynom: %.f | MSE train: %.3f | MSE test: %.3f | R2 train: %.3f | R2 test: %.3f)" %(p,i,j,k,l))
    
    
    
    
    
    
    
    
    
    
    
    
    