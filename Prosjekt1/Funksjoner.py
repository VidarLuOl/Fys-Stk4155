
import numpy as np #For FrankeFunction, MSE, R2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.linear_model as slm
from sklearn.utils import resample, shuffle
import matplotlib.pyplot as plt
from seaborn import lineplot
from random import choice

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def OLS(x, y, z, p, n, s, bootNumber, conf, lamda, prnt, plot, ridge, lasso):
    scaler = StandardScaler()

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
        
        beta = BetaFunc(XD_train_scaled, z_train, lamda, ridge, lasso) #Vi finner beta verdiene her
        #beta = (X^T * X)^-1 (X^T * y) 

        ztilde_train = XD_train_scaled @ beta
        ztilde_test = XD_test_scaled @ beta
        #y = X * beta

        MeanSETrain = MSE(z_train, ztilde_train)
        MeanSETest = MSE(z_test, ztilde_test)
        #1/n sum(y - model(y))^2

        plotMSETrain[i-1] = MeanSETrain
        plotMSETest[i-1] = MeanSETest

        R2Train_score = R2(z_train, ztilde_train)
        R2Test_score = R2(z_test, ztilde_test)
        #1- sum(y - model(y))^2/sum(y - mean(y))^2

        print(np.shape(XD_train_scaled), n)

        # beta_variance = Variance(XD_train_scaled, n)

        beta_variance = Variance(XD_train_scaled, np.shape(XD_train_scaled)[0])
        beta_ConfInt = ConfInt(conf, beta_variance, np.shape(XD_train_scaled)[0])
        # beta_ConfInt = ConfInt(conf, beta_variance, n)



        if(prnt == 1):
            print("Skalert og trent OLS")
            print("Grad = %i (p)" %i)
            print("Antall undersøkt = %i (n)" %n)
            print("MSE = %.6f" %MeanSETrain)
            print("R2Train = %.6f" %R2Train_score)
            print("R2Test = %.6f" %R2Test_score)
            print("")

    if(plot == 1):
        plt.plot(range(1,p+1), plotMSETrain, label="Train")
        plt.plot(range(1,p+1), plotMSETest, label="Test")
        plt.title("MSE of training and test set")
        plt.show()
        plt.title("Confidence intervall for the different betas")
        plt.errorbar(range(0,len(beta)), beta, beta_ConfInt, fmt="o")
        plt.show()

def Bootstrap(x, y, z, p, n, s, bootNumber, conf, lamda, prnt, plot, ridge, lasso):
    """
    Størrelse på ulike variabler
    XD = (n*n, antall betaer)
    bootXTrain = (bootSize, antall betaer)
    beta = (antall betaer, 1)
    ztilde_train = (bootSize, 1)
    beta_variance= (antall beta, )
    """
    scaler = StandardScaler()
    bootMSETrain = np.empty(bootNumber)
    bootMSETest = np.empty(bootNumber)

    plotMSETrain = np.zeros(p)
    plotMSETest = np.zeros(p)

    error = np.zeros(p)
    bias = np.zeros(p)
    vvariance = np.zeros(p)
    R2_score = np.zeros(bootNumber)

    for i in range(1, p+1):
        XD = Design_X(x, y, i) #Designmatrisen blir laget her, (n*n, antall beta) matrix

        XD_train, XD_test, z_train, z_test = train_test_split(XD, z.reshape(-1,1), test_size=s)

        scaler.fit(XD_train)
        XD_train_scaled = scaler.transform(XD_train)
        XD_test_scaled = scaler.transform(XD_test)

        ztilde_BootTrain = np.zeros((len(z_train), bootNumber))
        ztilde_BootTest = np.zeros((len(z_test), bootNumber))

        XD_train_scaled[:,0] = 1
        XD_test_scaled[:,0] = 1

        for j in range(0, bootNumber):
            bootXTrain, bootZTrain = resample(XD_train_scaled, z_train, replace=True)
            bootXTest, bootZTest = resample(XD_test_scaled, z_test, replace=True)

            beta = BetaFunc(bootXTrain, bootZTrain, lamda, ridge, lasso) #Vi finner beta verdiene her

            if(lasso):
                ztilde_train = beta.predict(bootXTrain)
                ztilde_test = beta.predict(bootXTest)
            else:
                ztilde_train = bootXTrain @ beta
                ztilde_test = bootXTest @ beta

            ztilde_BootTest[:,j] = np.ravel(ztilde_test)
            ztilde_BootTrain[:,j] = np.ravel(ztilde_train)

            bootMSETrain[j] = MSE(bootZTrain, ztilde_train)
            bootMSETest[j] = MSE(bootZTest, ztilde_test)

            R2_score[j] = R2(bootZTest, ztilde_test)

            beta_variance = Variance(bootXTrain, np.shape(bootXTrain)[0])
            beta_ConfInt = ConfInt(conf, beta_variance, np.shape(bootXTrain)[0])

        plotMSETrain[i-1] = np.mean(bootMSETrain)
        plotMSETest[i-1] = np.mean(bootMSETest)

        error[i-1] = np.mean(np.mean((z_test - ztilde_BootTest)**2, axis=1, keepdims=True))
        bias[i-1] = np.mean((z_test - np.mean(ztilde_BootTest, axis=1, keepdims=True))**2)
        vvariance[i-1] = np.mean(np.var(ztilde_BootTest, axis=1, keepdims=True))

        if(prnt == 1):
            print("Skalert og trent OLS")
            print("Grad = %i (p)" %i)
            print("Antall undersøkt = %i (n)" %n)
            print("MSE = %.6f" %plotMSETest[i-1])
            print("R2 = %.6f" %np.mean(R2_score))
            print("Error = ", error[i-1])
            print("Bias = ", bias[i-1])
            print("Variance = ", vvariance[i-1])
            print("")

    if(plot == 1):
        plt.plot(range(1,p+1), plotMSETrain, label="Train")
        plt.plot(range(1,p+1), plotMSETest, label="Test")
        plt.title("MSE of training and test set")
        plt.show()
        plt.plot(range(1,p+1), error, label="MSE")
        plt.plot(range(1,p+1), bias, label="Bias")
        plt.plot(range(1,p+1), vvariance, label="Variance")
        plt.legend()
        plt.show()
        if(lasso == False):
            plt.title("Confidence intervall for the different betas")
            plt.errorbar(range(0,len(beta)), beta, beta_ConfInt, fmt="o")
            plt.show()

def CV(x, y, z, p, n, cvAntall, conf, lamda, prnt, plot, ridge, lasso):
    scaler = StandardScaler()

    plotMSETrain = np.zeros(p)
    plotMSETest = np.zeros(p)

    for k in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]:
        lamda = k
        for i in range(1, p+1):
            XD = Design_X(x, y, i) #Designmatrisen blir laget her
            meanMSEVectorTrain = np.zeros(cvAntall)
            meanMSEVectorTest = np.zeros(cvAntall)

            XD_random, z_random = shuffle(XD, z)

            XDVector = np.array_split(XD_random, cvAntall)
            zVector = np.array_split(z_random, cvAntall)   #Splitter inn i like store grupper

            
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

                beta = BetaFunc(XD_train_scaled, z_train, lamda, ridge, lasso) 
        
                if(lasso):
                    ztilde_train = beta.predict(XD_train_scaled)
                    ztilde_test = beta.predict(XD_test_scaled)
                else:
                    ztilde_train = XD_train_scaled @ beta
                    ztilde_test = XD_test_scaled @ beta

                MeanSETrain = MSE(z_train, ztilde_train)
                MeanSETest = MSE(z_test, ztilde_test)

                meanMSEVectorTrain[j] = MeanSETrain
                meanMSEVectorTest[j] = MeanSETest

                R2_score = R2(z_train, ztilde_train)

                beta_variance = Variance(XD_train_scaled, np.shape(XD_train_scaled)[0])
                beta_ConfInt = ConfInt(conf, beta_variance, np.shape(XD_train_scaled)[0])

                if(prnt == 1):
                    print("Skalert og trent OLS")
                    print("Grad = %i (p)" %i)
                    print("Antall undersøkt = %i (n)" %n)
                    print("MSE = %.6f" %MeanSETrain)
                    print("R2 = %.6f" %R2_score)
                    print("")

            plotMSETrain[i-1] = np.mean(meanMSEVectorTrain)
            plotMSETest[i-1] = np.mean(meanMSEVectorTest)

        if(plot == 1):
            # plt.plot(range(1,p+1), plotMSETrain, label="Train")
            plt.plot(range(1,p+1), plotMSETest, label=("lambda = ", lamda))
            # plt.title("MSE of training and test set")
            # plt.show()
            # plt.title("Confidence intervall for the different betas")
            # if(lasso):
            #     plt.errorbar(range(0,len(beta.coef_)), beta.coef_, 1, fmt="o")
            # else:
            #     plt.errorbar(range(0,len(beta)), beta, beta_ConfInt, fmt="o")
            # plt.show()
    plt.legend()
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

def BetaFunc(X, z, lamda, ridge, lasso):
    if(ridge == True):
        return np.linalg.inv(X.T @ X + lamda*np.identity(len(X[0]))) @ (X.T @ z)
    elif(lasso == True):
        return slm.Lasso(alpha = lamda).fit(X, z)
    else:
        return np.linalg.inv(X.T @ X) @ (X.T @ z)

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
    return 1 - np.sum((y - ytilde) ** 2) / np.sum((y - np.mean(y)) ** 2)

def Variance(X, n):
    var = sum((X - np.mean(X))**2) * np.diag(np.linalg.pinv(X.T @ X))
    return var

def ConfInt(conf, variance, n):
    return conf * np.sqrt(variance)/n
"""______________________________________________________________________________________________"""