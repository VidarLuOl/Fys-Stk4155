
import numpy as np #For FrankeFunction, MSE, R2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
import matplotlib.pyplot as plt
# from seaborn import lineplot
from random import choice

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def OLS(x, y, z, p, n, s, conf, prnt, plot):
    scaler = StandardScaler(with_std=False)

    plotMSETrain = np.zeros(p)
    plotMSETest = np.zeros(p)
    

    for i in range(1, p+1):
        XD = Design_X(x, y, i) #Designmatrisen blir laget her

        XD_train, XD_test, z_train, z_test = train_test_split(XD, z.reshape(-1,1), test_size=s)

        scaler.fit(XD_train)
        XD_train_scaled = scaler.transform(XD_train)
        XD_test_scaled = scaler.transform(XD_test)

        XD_train_scaled[:,0] = 1
        XD_test_scaled[:,0] = 1
        
        beta = BetaFunc(XD_train_scaled, z_train) #Vi finner beta verdiene her

        ztilde_train = XD_train_scaled @ beta
        ztilde_test = XD_test_scaled @ beta

        MeanSETrain = MSE(z_train, ztilde_train)
        MeanSETest = MSE(z_test, ztilde_test)

        plotMSETrain[i-1] = MeanSETrain
        plotMSETest[i-1] = MeanSETest

        R2_score = R2(z_train, ztilde_train)

        beta_variance = Variance(XD_train_scaled, n)*np.diag(np.linalg.inv(XD_train_scaled.T @ XD_train_scaled))

        beta_std = ConfInt(conf, beta_variance, n)

        if(prnt == 1):
            print("Skalert og trent OLS")
            print("Grad = %i (p)" %i)
            print("Antall undersøkt = %i (n)" %n)
            print("MSE = %.6f" %MeanSETrain)
            print("R2 = %.6f" %R2_score)
            print("")

    if(plot == 1):
        plt.plot(range(0,p), plotMSETrain)
        plt.plot(range(0,p), plotMSETest)
        plt.title("MSE of training and test set")
        plt.show()
        plt.title("Confidence intervall for the different betas")
        plt.errorbar(range(0,len(beta)), beta, beta_std, fmt="o")
        plt.show()

def Bootstrap(x, y, z, p, n, s, bootNumber, bootSize, conf, prnt, plot):
    """
    Størrelse på ulike variabler
    XD = (n*n, antall betaer)
    bootXTrain = (bootSize, antall betaer)
    beta = (antall betaer, 1)
    ztilde_train = (bootSize, 1)
    beta_variance= (antall beta, )
    """
    scaler = StandardScaler(with_std=False)
    bootMSE = np.empty(p)
    

    plotMSETrain = np.zeros(p)
    plotMSETest = np.zeros(p)
        

    for i in range(1, p+1):
        XD = Design_X(x, y, i) #Designmatrisen blir laget her, (n*n, antall beta) matrix

        XD_train, XD_test, z_train, z_test = train_test_split(XD, z.reshape(-1,1), test_size=s)

        scaler.fit(XD_train)
        XD_train_scaled = scaler.transform(XD_train)
        XD_test_scaled = scaler.transform(XD_test)

        XD_train_scaled[:,0] = 1
        XD_test_scaled[:,0] = 1

        for j in range(0, bootNumber):
            bootXTrain, bootYTrain = resample(XD_train_scaled, z_train, n_samples=bootSize, replace=True)
            bootXTest, bootYTest = resample(XD_test_scaled, z_test, n_samples=bootSize, replace=True)
            print(np.shape(bootXTrain))

            beta = BetaFunc(bootXTrain, bootYTrain) #Vi finner beta verdiene her

            ztilde_train = bootXTrain @ beta
            ztilde_test = bootXTest @ beta

            MeanSETrain = MSE(bootYTrain, ztilde_train)
            MeanSETest = MSE(bootYTest, ztilde_test)

            plotMSETrain[i-1] = MeanSETrain
            plotMSETest[i-1] = MeanSETest

            R2_score = R2(z_train, bootYTrain)

            beta_variance = Variance(bootXTrain, n)*np.diag(np.linalg.inv(bootXTrain.T @ bootXTrain))

            beta_std = ConfInt(conf, beta_variance, n)
    
            if(prnt == 1):
                print("Skalert og trent OLS")
                print("Grad = %i (p)" %i)
                print("Antall undersøkt = %i (n)" %n)
                print("MSE = %.6f" %MeanSETrain)
                print("R2 = %.6f" %R2_score)
                print("")
        # bootMSE[i-1] = np.mean( np.mean((z_test - ztilde_train.reshape(100,8))**2, axis=1, keepdims=True))
        # print(np.shape(ztilde_train.reshape(100,8)))
        # print(bootMSE)

    if(plot == 1):
        plt.plot(range(1,p+1), plotMSETrain)
        plt.plot(range(1,p+1), plotMSETest)
        plt.title("MSE of training and test set")
        plt.show()
        plt.title("Confidence intervall for the different betas")
        plt.errorbar(range(0,len(beta)), beta, beta_std, fmt="o")
        plt.show()

def CV(x, y, z, p, n, s, cvAntall, conf, prnt, plot):
    scaler = StandardScaler(with_std=False)

    plotMSETrain = np.zeros(p)
    plotMSETest = np.zeros(p)
    

    for i in range(1, p+1):
        XD = Design_X(x, y, i) #Designmatrisen blir laget her

        bestCVXD = np.zeros(int(n*n*0.75))
        bestCVz = np.zeros(int(n*n*0.25))
        score = 2

        XD_random, z_random = shuffle(XD, z)

        XDVector = np.array_split(XD_random, cvAntall)
        zVector = np.array_split(z_random, cvAntall)   #Splitter inn i like store grupper
        for k in range(cvAntall):
            XDVector[k] = XDVector[k].tolist()
            zVector[k] = zVector[k].tolist()

        
        for j in range(0, cvAntall):
            XD_train = XDVector[:j] + XDVector[j+1:] 
            XD_train = [j for i in XD_train for j in i]
            XD_test = XDVector[j]
            z_train = zVector[:j] + zVector[j+1:]
            z_train = [j for i in z_train for j in i]
            z_test = zVector[j]

            scaler.fit(XD_train)

            XD_train_scaled = scaler.transform(XD_train)
            XD_test_scaled = scaler.transform(XD_test)

            XD_train_scaled[:,0] = 1
            XD_test_scaled[:,0] = 1

            beta = BetaFunc(XD_train_scaled, z_train)

            ztilde_train = XD_train_scaled @ beta
            ztilde_test = XD_test_scaled @ beta

            MeanSETrain = MSE(z_train, ztilde_train)
            MeanSETest = MSE(z_test, ztilde_test)

            plotMSETrain[i-1] = MeanSETrain
            if(MeanSETest < score):
                plotMSETest[i-1] = MeanSETest
                score = MeanSETest

            R2_score = R2(z_train, ztilde_train)
            print(plotMSETest)

            beta_variance = Variance(XD_train_scaled, n)*np.diag(np.linalg.inv(XD_train_scaled.T @ XD_train_scaled))

            beta_std = ConfInt(conf, beta_variance, n)

            if(prnt == 1):
                print("Skalert og trent OLS")
                print("Grad = %i (p)" %i)
                print("Antall undersøkt = %i (n)" %n)
                print("MSE = %.6f" %MeanSETrain)
                print("R2 = %.6f" %R2_score)
                print("")

    if(plot == 1):
        # plt.plot(range(0,p), plotMSETrain)
        plt.plot(range(0,p), plotMSETest)
        plt.title("MSE of training and test set")
        plt.show()
        plt.title("Confidence intervall for the different betas")
        plt.errorbar(range(0,len(beta)), beta, beta_std, fmt="o")
        plt.show()




"""___________________________________Mindre Funksjoner________________________________"""
def Design_X(x, y, p):
    if len(x.shape)>1:
        x = np.ravel(x)
        y = np.ravel(y)
    
    r = len(x)                  #Antall rader
    c = int((p+1)*(p+2)/2)      #Antall kolonner
    X = np.ones((r, c))         #Lager en matrise hvor alle verdiene er 1

    for i in range(1, p+1):             #looping over the degrees but we have 21 elements
        q = int(i*(i+1)/2)              #Gjør det slik at vi 
        for k in range(i+1):            #this loop makes sure we dont fill the same values in X over and over
            X[:,q+k] = x**(i-k)*y**k    #using the binomial theorem to get the right exponents
    return X

def BetaFunc(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

def MSE(y, ytilde):
    #MSE er når vi skal sjekke hvor nøyaktig modellen vi har lagd er iforhold til punktene vi har fått inn
    #Jo nærmere 0 jo bedre er MSE, hvis den er 0 så er den mest sannsynlig overfittet pga at normalt vil støy ødellege.
    s = 0
    n = np.size(y)
    for i in range(0,n-1):
        s += (y[i] - ytilde[i])**2
    return s/n

def R2(y, ytilde):
    #R2 er en skala mellom 0 og 1, hvor jo nærmere 1 enn er jo bedre er det
    u = 0
    l = 0
    m = np.mean(y)
    for i in range(0, np.size(y)-1):
        u += (y[i] - ytilde[i])**2
        l += (y[i] - m)**2
    return 1 - u/l

def Variance(X, n):
    var = (1/n)*sum((X - np.mean(X))**2)
    return var

def ConfInt(conf, variance, n):
    return conf*(np.sqrt(variance)/np.sqrt(n)) #1.96*(sigma/sqrt(n))
"""______________________________________________________________________________________________"""