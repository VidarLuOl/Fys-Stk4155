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
p = 11          #Grad av polynom
s = 0.3         #Hvor stor del av dataen som skal være test
conf = 1.96     #Confidence intervall, 95% her


Opg = 1         #Hvilken Opg som skal kjøre
prnt = 1        #Om du vil printe ut resultater. 0=nei, 1=ja
plot = 1        #Om du vil plotte ut resultater. 0=nei, 1=ja

"""_________________________________________________________"""


# Make data.
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)
x, y = np.meshgrid(x,y)

noise_full = noise*np.random.randn(n, n)


def Opg1():
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    OLS(x, y, z, p, n, s, conf, prnt, plot)

def Opg2():
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    Bootstrap(x, y, z, p, n, s, conf, prnt, plot)



if __name__ == "__main__":
    if(Opg == 1):
        Opg1()
    elif(Opg == 2):
        Opg2()
    else:
        print("Du må velge en oppgave!")
