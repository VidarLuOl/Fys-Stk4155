import numpy as np
from Functions import FrankeFunction, OLS, boot, bootstrap
import matplotlib.pyplot as plt
from scipy.stats import norm



np.random.seed(20)

N = 25
dt = float(1/N)
x = np.arange(0, 1, dt)
y = np.arange(0, 1, dt)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x,y)
z_noise = FrankeFunction(x,y) + 0.2*np.random.randn(len(x),len(x))
order = 20
p = 15
        
data = OLS(x,y,z_noise,p)
X_train_scaled = data[-4]

"""
n_samples = 1000
t = boot(X_train_scaled, n_samples)
n, binsboot, patches = plt.hist(t, 50, density=True, facecolor='red', alpha=0.75)

yb = norm.pdf(binsboot, np.mean(t), np.std(t))
lt = plt.plot(binsboot, yb, 'b', linewidth=1)
plt.xlabel('x')
plt.ylabel('Probability')
plt.grid(True)
plt.show()
"""

n_bootstraps = 250
maxdegree = 10
    
err,bi,var,polydeg = bootstrap(x,y,z_noise, maxdegree, n_bootstraps)

plt.figure()
plt.plot(polydeg, err, label='Error')
plt.plot(polydeg, bi, label='bias')
plt.plot(polydeg, var, label='Variance')
plt.legend()
plt.show()





