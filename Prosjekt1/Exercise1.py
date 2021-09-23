import numpy as np
from Functions import FrankeFunction, OLS, plot
import matplotlib.pyplot as plt

np.random.seed(2021)

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x,y)
z_noise = FrankeFunction(x,y) + 0.05*np.random.randn(20,20)


P = 5
polynomials = [_ for _ in range(1,P+1)]

data = OLS(x,y,z_noise,polynomials)

"""
print("Normal data")
for p,i,j,k,l in zip(polynomials,data[0], data[2], data[1], data[3]):
    print("Polynom: %.f | MSE train: %.3f | MSE test: %.3f | R2 train: %.3f | R2 test: %.3f)" %(p,i,j,k,l))
    
    
print("Scaled data")
for p,i,j,k,l in zip(polynomials,data[5], data[7], data[6], data[8]):
    print("Polynom: %.f | MSE train: %.3f | MSE test: %.3f | R2 train: %.3f | R2 test: %.3f)" %(p,i,j,k,l))
"""


"""
#plot(polynomials, sigma, "sigma", "std")

lower_mu = []
upper_mu = []


for _ in range(len(sigma)):
    lower_mu.append(mu[_] - 1.96*sigma[_]/(len(sigma)))
    upper_mu.append(mu[_] + 1.96*sigma[_]/(len(sigma)))
"""






plt.figure()
x_axis = np.linspace(0,len(beta5),len(beta5))
plt.errorbar(x_axis, beta5, sigma, fmt="o")


"""
plot(polynomials,data[0], "MSE train", "mse score")
plot(polynomials,data[1], "R2 train", "r2 score")
plot(polynomials,data[2], "MSE test", "mse score")
plot(polynomials,data[3], "R2 test", "r2 score")


plot(polynomials,data[5], "MSE train scaled", "mse score")
plot(polynomials,data[6], "R2 train scaled", "r2 score")
plot(polynomials,data[7], "MSE test scaled", "mse score")
plot(polynomials,data[8], "R2 test scaled", "r2 score")        
"""