from Funksjoner import *

#For FrankeFunction
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

from imageio import imread

# seed = 31415


"""_________________________Variabler_______________________"""
n = 20          #Antall punkter i x- og y-retning på modellen
noise = 0.1     #Hvor mye støy påvirker modellen 
p = 10          #Grad av polynom
s = 0.3         #Hvor stor del av dataen som skal være test
conf = 1.96     #Confidence intervall, 95% her
lamda = 1e-10   #Swag master 3000 verdier lizm

"Opg 2"
bootNumber = 100  #Antall bootstrap 

"Opg 3"
cvAntall = 10

"Hvilken oppgave vil du kjøre"
Opg = 6        #Hvilken Opg som skal kjøre, 1 = OLS, 2 = Bootstrap, 3 = CV
ridge = False   #Om du vil inkludere Ridge Regression på Frankefunksjonen
lasso = False   #Om du vil inkludere Lasso Regression på Frankefunksjonen
prnt = 1        #Om du vil printe ut resultater. 0=nei, 1=ja
plot = 1        #Om du vil plotte ut resultater. 0=nei, 1=ja

"""_________________________________________________________"""


# Make data.
# x = np.arange(0, 1, 1/n)
# y = np.arange(0, 1, 1/n)

# x, y = np.meshgrid(x,y)

# noise_full = noise*np.random.randn(n, n)

terrain = imread("SRTM_data_Norway_1.tif")

terrain = terrain[400:550,1400:1600]

x = np.arange(0, 1, 1/len(terrain[0]))
y = np.arange(0, 1, 1/len(terrain[:,0]))
# x = np.arange(0, len(terrain[0]))
# y = np.arange(0, len(terrain[:,0]))

x,y = np.meshgrid(x,y)

noise_full = noise*np.random.randn(len(terrain[:,0]), len(terrain[0]))

# len(terrain[0]) = 1801
# len(terrain[:,0]) = 3601
# plt.figure()
# plt.title("Terrain over Norway 1")
# # plt.imshow(terrain, cmap="gray")
# plt.contour(x, y, terrain, 3)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# print(terrain)




def Opg1():
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    OLS(x, y, z, p, n, s, conf, lamda, prnt, plot, ridge, lasso)

def Opg2():
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    Bootstrap(x, y, z, p, n, s, bootNumber, conf, lamda, prnt, plot, ridge, lasso)

def Opg3():
    z = np.ravel(FrankeFunction(x, y) + noise_full)

    CV(x, y, z, p, n, cvAntall, conf, lamda, prnt, plot, ridge, lasso)

def Opg6():
    z = np.ravel(terrain + noise_full)
    # OLS(x, y, z, p, n, s, conf, lamda, prnt, plot, ridge, lasso)
    # Bootstrap(x, y, z, p, n, s, bootNumber, conf, lamda, prnt, plot, ridge, lasso)
    CV(x, y, z, p, n, cvAntall, conf, lamda, prnt, plot, ridge, lasso)

if __name__ == "__main__":
    if(Opg == 1):
        Opg1()
    elif(Opg == 2):
        Opg2()
    elif(Opg == 3):
        Opg3()
    elif(Opg == 6):
        Opg6()
    else:
        print("Du må velge en oppgave!")

