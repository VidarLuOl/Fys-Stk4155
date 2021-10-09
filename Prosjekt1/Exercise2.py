from Functions import datapoints, OLS, bootstrap_OLS
import matplotlib.pyplot as plt






x,y,z = datapoints()

order = 20
p = 15
        
data = OLS(x,y,z,p)
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
maxdegree = 15
    
err,bi,var,polydeg = bootstrap_OLS(x,y,z, maxdegree, n_bootstraps)

plt.figure()
plt.semilogy(polydeg, err, label='Error')
plt.semilogy(polydeg, bi, label='bias')
plt.semilogy(polydeg, var, label='Variance')
plt.legend()
plt.show()





