import numpy as np #For FrankeFunction, MSE, R2

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def OLS(x, y, z, p, n):
    XD = Design_X(x, y, p) #Designmatrisen blir laget her

    z = np.ravel(z)  #Vi er nødt til å gjøre om y til en vektor ettersom det er en matrise før.

    beta = betaFunc(XD, z) #Vi finner beta verdiene her

    ztilde = XD @ beta
    ztilde_plot = np.reshape(ztilde, (n,n))

    MeanSE = MSE(z, ztilde)

    R2_score = R2(z, ztilde)
    return MeanSE, R2_score, ztilde_plot

"""___________________________________OSL FUNKSJONER________________________________"""
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

def betaFunc(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

def MSE(y, ytilde):
    #MSE er når vi skal sjekke hvor nøyaktig modellen vi har lagd er iforhold til punktene vi har fått inn
    #Jo nærmere 0 jo bedre er MSE, hvis den er 0 så er den mest sannsynlig overfittet pga at normalt vil støy ødellege.
    s = 0
    n = np.size(y)
    for i in range(0,n-1):
        s += (y[i] - ytilde[i])**2
    return (1/n) * s

def R2(y, ytilde):
    #R2 er en skala mellom 0 og 1, hvor jo nærmere 1 enn er jo bedre er det
    u = 0
    l = 0
    m = np.mean(y)
    for i in range(0, np.size(y)-1):
        u += (y[i] - ytilde[i])**2
        l += (y[i] - m)**2
    return 1 - u/l
"""______________________________________________________________________________________________"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def Scaled_OSL(x, y, z, p, n):
    scaler = StandardScaler()
    
    XD = Design_X(x, y, p) #Designmatrisen blir laget her

    scaler.fit(XD)
    XD = scaler.transform(XD)
    XD[:, 0] = 1

    

    # z = np.ravel(z)  #Vi er nødt til å gjøre om y til en vektor ettersom det er en matrise før.

    # beta = betaFunc(XD, z) #Vi finner beta verdiene her

    # ztilde = XD @ beta
    # ztilde_plot = np.reshape(ztilde, (n,n))

    # MeanSE = MSE(z, ztilde)

    # R2_score = R2(z, ztilde)
    return "hei"