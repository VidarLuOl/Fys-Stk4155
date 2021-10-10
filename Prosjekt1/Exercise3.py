import numpy as np
from Functions import datapoints, OLS, bootstrapOLS, create_X, beta
import matplotlib.pyplot as plt
import random as random
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


x,y,z = datapoints()

def OLSkFold(x,y,z, maxdegree, kfold):
    #mse_train = np.empty((kfold, maxdegree))
    #mse_test = np.empty((kfold, maxdegree))
    #r2_train = np.empty((kfold, maxdegree))
    #r2_test = np.empty((kfold, maxdegree)) 
    
    mse_train = np.empty((maxdegree, kfold))
    mse_test = np.empty((maxdegree, kfold))
    r2_train = np.empty((maxdegree, kfold))
    r2_test = np.empty((maxdegree, kfold))      
    
    
    for degree in range(1,maxdegree):
        X = create_X(x,y,degree)
        z_data = np.ravel(z)
        X_train, z_train = shuffle(X, z_data)
        
        
        x_data = np.array_split(X_train, kfolds)
        z_data = np.array_split(z_train, kfolds)
        
    
        for i in range(0,len(x_data)):    
            X_test = x_data[i]
            X_train = x_data.copy()
            X_train.pop(i)
            X_train = np.concatenate(X_train, axis=0)
            
            z_test = z_data[i]
            z_train = z_data.copy()
            z_train.pop(i)
            z_train = np.concatenate(z_train, axis=0)
            
            scaler = StandardScaler()
            scaler.fit(X_train)
            
            X_train_scale = scaler.transform(X_train)
            X_test_scale = scaler.transform(X_test)
        
            X_train_scale[:,0] = 1
            X_test_scale[:,0] = 1
            
            z_mean_train = np.mean(z_train)
            z_train = (z_train - z_mean_train)/np.std(z_train)
            
            z_mean_test = np.mean(z_test)
            z_train = (z_train - z_mean_test)/np.std(z_test)
            
            coefs = beta(X_train_scale, z_train)
            
            z_fit = X_train_scale.dot(coefs)
            z_pred = X_test_scale.dot(coefs)
            
            
            MSE_train = mean_squared_error(z_fit, z_train)
            MSE_test = mean_squared_error(z_pred, z_test)
            
            R2_train = r2_score(z_fit, z_train)
            R2_test = r2_score(z_pred, z_test)

            mse_train[degree][i] = MSE_train
            mse_test[degree][i] = MSE_test
            r2_train[degree][i] = R2_train
            r2_test[degree][i] = R2_test    
            
    return mse_train, mse_test, r2_train, r2_test

maxdegree = 12
kfolds = 5


msetrain, msetest, r2train, r2test = OLSkFold(x,y,z,maxdegree, kfolds)



msetrain = [np.mean(msetrain[_]) for _ in range(maxdegree)]
msetest = [np.mean(msetest[_]) for _ in range(maxdegree)]
r2train = [np.mean(r2train[_]) for _ in range(maxdegree)]
r2test = [np.mean(r2test[_]) for _ in range(maxdegree)]

polynoms = np.linspace(1,maxdegree, maxdegree)

plt.figure()         
plt.title("MSE kFold OLS") 
plt.plot(polynoms,msetrain, label = "MSE train", color = "r")                
plt.plot(polynoms,msetest, label = "MSE test", color = "b")
plt.plot(polynoms,msetrain, "o", color = "r")                
plt.plot(polynoms,msetest, "o", color = "b")  
plt.xlabel("order")
plt.ylabel("mse")
plt.legend()
plt.show()  
    
plt.figure()         
plt.title("R2 kFold OLS") 
plt.plot(polynoms,r2train, label = "R2 train", color = "r")                
plt.plot(polynoms,r2test, label = "R2 test", color = "b")
plt.plot(polynoms,r2train, "o", color = "r")                
plt.plot(polynoms,r2test, "o", color = "b")  
plt.xlabel("order")
plt.ylabel("R2")
plt.legend()
plt.show() 


    