from Functions import datapoints, OLS, bootstrapOLS, boot
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


x,y,z = datapoints()

order = 20
p = 15
        
data = OLS(x,y,z,p)
X_train_scaled = data[-4]


n_samples = 500
t = boot(X_train_scaled, n_samples)
n, binsboot, patches = plt.hist(t, 50, density=True, facecolor='red', alpha=0.75)

yb = norm.pdf(binsboot, np.mean(t), np.std(t))
lt = plt.plot(binsboot, yb, 'b', linewidth=1)
plt.xlabel('x')
plt.ylabel('Probability')
plt.grid(True)
plt.show()


n_bootstraps = 625
maxdegree = 12
    
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





