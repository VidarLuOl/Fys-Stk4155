from Funksjoner import *

#For FrankeFunction
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

# seed = 31415


"""_________________________Variabler_______________________"""
n = 20          #Antall punkter i x- og y-retning på modellen
noise = 0.1     #Hvor mye støy påvirker modellen 
p = 10           #Grad av polynom
s = 0.3         #Hvor stor del av dataen som skal være test
conf = 1.96     #Confidence intervall, 95% her

"Opg 2"
bootNumber = 50  #Antall bootstrap 
bootSize = 800  #Hvor mange punkter i hver bootstrap, minimum (2(1/2*(n-7)**2 + 1/2(n-7)) + 97) hvor n er antall punkter i XD

"Opg 3"
cvAntall = 10

Opg = 2         #Hvilken Opg som skal kjøre
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

    Bootstrap(x, y, z, p, n, s, bootNumber, bootSize, conf, prnt, plot)

def Opg3():
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    CV(x, y, z, p, n, cvAntall, conf, prnt, plot)



if __name__ == "__main__":
    if(Opg == 1):
        Opg1()
    elif(Opg == 2):
        Opg2()
    elif(Opg == 3):
        Opg3()
    else:
        print("Du må velge en oppgave!")
