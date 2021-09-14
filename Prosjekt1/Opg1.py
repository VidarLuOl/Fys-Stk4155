from Funksjoner import *

#For FrankeFunction
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed


"""_________________________Variabler_______________________"""
n = 20          #Antall punkter i x- og y-retning på modellen
noise = 0.1     #Hvor mye støy påvirker modellen 
p = 5           #Grad av polynom
s = 0.3         #Hvor vi skal splitte dataen
Opg = 1         #Hvilken Opg som skal kjøre

"""_________________________________________________________"""


# Make data.
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)
x, y = np.meshgrid(x,y)

noise_full = noise*np.random.randn(n, n)


def Opg1():
     z = np.ravel(FrankeFunction(x, y) + noise_full)

    MeanSE, R2_score, ztilde_plot = OLS(x, y, z, p, n)

    # print("Normal OLS")
    # print("Grad = %i" %p)
    # print("Antall undersøkt = %i" %n)
    # print("MSE = %.6f" %MeanSE)
    # print("R2 = %.6f" %R2_score)

    MeanSE, R2_score, ztilde_train, ztilde_test = Scaled_OSL(x, y, z, p, s)


    # print("Skalert og trent OLS")
    # print("Grad = %i" %p)
    # print("Antall undersøkt = %i" %n)
    # print("MSE = %.6f" %MeanSE)
    # print("R2 = %.6f" %R2_score)


if __name__ == "__main__":
    if(Opg == 1):
        Opg1()
    elif(Opg == 2):
        Opg2()