import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, OneHotEncoder, OrdinalEncoder 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def str_arr(arr):
    nx, ny = arr.shape
    strs = []
    for i in range(ny):
        tipos = list(map(type,data[:,i]))
        dic = {i:tipos.count(i) for i in tipos}
        valores = dic.values()
        clave = dic.keys()
        if type('a') in clave:
            strs.append(i)
    nums = list(set(range(ny)) - set(strs))        
    return strs, nums

def nan_ave(arr,cols):
    df = pd.DataFrame(arr)
    nx, ny = arr.shape
    ave = []
    for i in cols:
        vals = []
        for j in range(nx):
            if np.isnan(arr[j,i]) == False:
                vals.append(arr[j,i])       
        ave = sum(vals)/len(vals)
        df[i] = df[i].replace(np.nan,ave)
    return df.values

def f_acc(ytrue,ypred):
    nn = len(ypred)
    if nn == len(ytrue):
        acc = 0
        for i in range(nn):
            acc += abs(ypred[i]-ytrue[i])/(ytrue[i] + 1E-10)
        return acc/nn    
    else: 
        raise ValueError("Please introduce arrays of the same dimensions")     

def print_out(arr1,nom1,arr2,nom2,nom):
    df = pd.DataFrame(zip(arr1,arr2),columns = [nom1,nom2])
    df.to_csv("results_"+str(nom)+'.csv',index=False)

dataset = pd.read_csv("train.csv")
Xdat = pd.read_csv("test.csv")
nx, ny = dataset.shape
missing = Xdat.isnull().sum()
missing = missing[missing > 0]#(ny-1)*0.66]

print(Xdat.shape)
Xdat.drop(missing.index,axis=1,inplace=True)
#Xdat.dropna(axis=1, inplace=True)
print(Xdat.shape)
print(nada)
#dataset.drop(missing.index,axis=1,inplace=True)
#Xdat.drop(missing.index,axis=1,inplace=True)
data = dataset.values
X = data[:,:-1]
Y = list(data[:,-1])

               
Xtr, Xtt, Ytr, Ytt = train_test_split(X,Y,test_size=0.2, random_state=0)
strs, nums = str_arr(X)
Xtr[:,strs] = Xtr[:,strs].astype(str)
Xtt[:,strs] = Xtt[:,strs].astype(str)
ord_enc = OrdinalEncoder() #OneHotEncoder()#
Xtr[:,strs], Xtt[:,strs] = ord_enc.fit_transform(Xtr[:,strs]), ord_enc.fit_transform(Xtt[:,strs])
Xtr, Xtt = nan_ave(Xtr,nums), nan_ave(Xtt,nums)
# ------ Machine learning modeling --------# 
model = RandomForestRegressor(random_state=0)
model.fit(Xtr,Ytr)
ysol = model.predict(Xtt)
err = f_acc(Ytt,ysol)
print("Average error in the prediction: "+str(round(err*100,3))+"%")

print_out(Ytt,"Value",ysol,"Prediction","train")




