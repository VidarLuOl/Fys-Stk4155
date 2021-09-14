from Funksjoner import *

#For FrankeFunction
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

#Andre imports


"""_________________________Variabler_______________________"""
n = 5          #Antall punkter i x- og y-retning på modellen
noise = 0.1     #Hvor mye støy påvirker modellen 
p = 2           #Grad av polynom

"""_________________________________________________________"""


fig = plt.figure()
ax = fig.gca(projection="3d")

# Make data.
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)
x, y = np.meshgrid(x,y)

noise_full = noise*np.random.randn(n, n)


if __name__ == "__main__":
    z = FrankeFunction(x, y) + noise_full

    MeanSE, R2_score, ztilde_plot = OLS(x, y, z, p, n)

    # print("Grad = %i" %p)
    # print("MSE = %.6f" %MeanSE)
    # print("R2 = %.6f" %R2_score)

    # # Plot the surface.
    # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # surf = ax.plot_surface(x, y, ztilde_plot, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # # Customize the z axis.
    # ax.set_zlim(-0.10, 1.40)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    """Fra herfra så prøver vi å skalere punktene"""
    Scaled = Scaled_OSL(x, y, z, p, n)

