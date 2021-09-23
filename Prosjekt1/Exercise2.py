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



    
testlist = [data[0], data[2]]

plt.figure()
for y in testlist:
    plt.plot(polynomials, y)
plt.show()