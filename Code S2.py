#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.linear_model import LinearRegression

def import_Excel_data(file_name,sheet_name):
    Cpx_data = pd.read_excel(file_name,sheet_name=sheet_name,usecols="E,F,R,AF:AQ,AT,AX,AY,BA:BD",skiprows=1)
    #P,T,H2O,Ti,Al,Cr,Fe,Mn,Mg,Ca,Na,K,AlT,AlVI,FeIII,Jd,DiHd,EnFs,MgM2,FeM1,FeM2,a_En
    #0 1  2  3  4  5  6  7  8  9 10 11 12  13    14   15  16  17    18   19   20   21
    Cpx_data = Cpx_data.fillna(0)
    Cpx_data.columns = [c.replace('.1','') for c in Cpx_data.columns]
    
    return Cpx_data

Data = import_Excel_data('Table S1.xlsx','Cali&Validation data')
Data = np.array(Data.values,dtype=np.float32)

train_32b_loss = np.array([[]])
test_32b_loss = np.array([[]])
train_32dH_loss = np.array([[]])
test_32dH_loss = np.array([[]])
train_32a_loss = np.array([[]])
test_32a_loss = np.array([[]])
train_N_loss = np.array([[]])
test_N_loss = np.array([[]])
train_NT_loss = np.array([[]])
test_NT_loss = np.array([[]])

train_32b_loss_iter = np.array([[]])
test_32b_loss_iter = np.array([[]])
train_32dH_loss_iter = np.array([[]])
test_32dH_loss_iter = np.array([[]])
train_32a_loss_iter = np.array([[]])
test_32a_loss_iter = np.array([[]])

#define loss function
def criterion(X,Y):
    Loss = np.array([[np.mean((X-Y)*(X-Y))]])
    return Loss

#define P32b model
def Bar32b1(X):
    T = X[:,[1]]
    H2O = X[:,[2]]
    AlVI = X[:,[13]]
    Fe = X[:,[6]]
    K = X[:,[11]]
    Jd = X[:,[15]]
    DiHd = X[:,[16]]
    Al = X[:,[4]]
    FeM2 = X[:,[20]]
    MgM2 = X[:,[18]]
    Y = np.concatenate((T,np.log(T),H2O,AlVI,Fe,K,Jd,DiHd,np.log(Jd),Al*Al,FeM2*FeM2,MgM2*MgM2,DiHd*DiHd),1)
    return Y

#P32b model used for iteration
def Bar32b2(X):
    H2O = X[:,[2]]
    AlVI = X[:,[13]]
    Fe = X[:,[6]]
    K = X[:,[11]]
    Jd = X[:,[15]]
    DiHd = X[:,[16]]
    Al = X[:,[4]]
    FeM2 = X[:,[20]]
    MgM2 = X[:,[18]]
    Y = np.concatenate((H2O,AlVI,Fe,K,Jd,DiHd,np.log(Jd),Al*Al,FeM2*FeM2,MgM2*MgM2,DiHd*DiHd),1)
    return Y

#define P32dH model
def K32dH(X):
    Ti = X[:,[3]]
    Fe = X[:,[6]]
    A = X[:,[4]]+X[:,[5]]-X[:,[10]]-X[:,[11]]
    a_En = X[:,[21]]
    B = np.log(a_En)*np.log(a_En)
    H2O = X[:,[2]]
    Y = np.concatenate((Ti,Fe,A,B,H2O),1)
    return Y

#define P32a model
def Bar32a1(X):
    T = X[:,[1]]
    Mg = X[:,[8]]
    Na = X[:,[10]]
    DiHd = X[:,[16]]
    EnFs = X[:,[17]]
    AlVI = X[:,[13]]
    Y = np.concatenate((T,np.log(T),Mg,Na,DiHd,AlVI,DiHd*DiHd,EnFs*EnFs),1)
    return Y

#P32a model used for iteration
def Bar32a2(X):
    Mg = X[:,[8]]
    Na = X[:,[10]]
    DiHd = X[:,[16]]
    EnFs = X[:,[17]]
    AlVI = X[:,[13]]
    Y = np.concatenate((Mg,Na,DiHd,AlVI,DiHd*DiHd,EnFs*EnFs),1)
    return Y

#define Nimis' model
def BarN(X):
    Ti = X[:,[3]]
    Cr = X[:,[5]]
    Mn = X[:,[7]]
    Ca = X[:,[9]]
    Na = X[:,[10]]
    AlT = X[:,[12]]
    AlM1 = X[:,[13]]
    FeIII = X[:,[14]]
    MgM2 = X[:,[18]]
    FeM1 = X[:,[19]]
    FeM2 = X[:,[20]]
    Y = np.concatenate((AlT,FeM1,FeIII,AlM1,Ti,Cr,Ca,Na,MgM2,FeM2,Mn,MgM2*MgM2,FeM2*FeM2),1)
    return Y

#define new thermometric model
def NewT(X):
    Ti = X[:,[3]]
    Al = X[:,[4]]
    Cr = X[:,[5]]
    Fe = X[:,[6]]
    Mn = X[:,[7]]
    Mg = X[:,[8]]
    Ca = X[:,[9]]
    #K = X[:,[11]]
    AlVI = X[:,[13]]
    FeII = X[:,[6]]-X[:,[14]]
    H2O = X[:,[2]]
    NLT = 1.4911*AlVI/(1.4911*AlVI+7.7083*Ti+1.1672*Cr+1.0687*Fe+0.2787*Mn-0.0627*Mg)
    Y = np.concatenate((NLT,Ti,Al,Mn,Mg,Ca,FeII,H2O),1)
    return Y

#repeat 100 times of splitting, calibrating and validating
for j in range(100):
    np.random.seed(j)

    for i in range (7):
        if i==0:
            data = Data[Data[:,0] == 0.001,:]   #extract data of P=0.001kbar
            np.random.shuffle(data)   #randomly sort
            a = math.floor(0.8*len(data))
            b = math.floor(len(data))
            Train_data = data[0:a,:]#extract the first 80% as calibration dataset
            Test_data = data[a:b,:]#the remaining data as validation dataset
        else:
            idx = np.where((Data[:,0]>(2*i-2+0.00103))&(Data[:,0]<=2*i))#extract data of 2i-2<P<=2i,except 0.001kbar data
            data = Data[idx]
            np.random.shuffle(data)   #randomly sort
            a = math.floor(0.8*len(data))
            b = math.floor(len(data))
            Train_data = np.concatenate((Train_data,data[0:a,:]),0)
            Test_data = np.concatenate((Test_data,data[a:b,:]),0)
    
    #Train model P32b
    Train_32b_x = Bar32b1(Train_data)
    Train_P = Train_data[:,[0]]
    Test_32b_x = Bar32b1(Test_data)
    Test_P = Test_data[:,[0]]
    
    Cpxbar32b = LinearRegression()
    Cpxbar32b.fit(Train_32b_x,Train_P)
    Train_32b_pred = Cpxbar32b.predict(Train_32b_x)
    loss_train = criterion(Train_32b_pred,Train_P)
    
    Test_32b_pred = Cpxbar32b.predict(Test_32b_x)
    loss_test = criterion(Test_32b_pred,Test_P)
    
    train_32b_loss = np.concatenate((train_32b_loss,loss_train),1)
    test_32b_loss = np.concatenate((test_32b_loss,loss_test),1)
    
    #Train model P32dH
    Train_32dH_x = K32dH(Train_data)
    Train_T = Train_data[:,[1]]
    Train_32dH_target = (93100+544*Train_data[:,[0]])/Train_T
    Test_32dH_x = K32dH(Test_data)
    Test_T = Test_data[:,[1]]
    
    Cpxtem32dH = LinearRegression()
    Cpxtem32dH.fit(Train_32dH_x,Train_32dH_target)
    Train_32dH_pred = (93100+544*Train_data[:,[0]])/Cpxtem32dH.predict(Train_32dH_x)
    loss_train = criterion(Train_32dH_pred,Train_T)
    
    Test_32dH_pred = (93100+544*Test_data[:,[0]])/Cpxtem32dH.predict(Test_32dH_x)
    loss_test = criterion(Test_32dH_pred,Test_T)
    
    train_32dH_loss = np.concatenate((train_32dH_loss,loss_train),1)
    test_32dH_loss = np.concatenate((test_32dH_loss,loss_test),1)
    
    #Train model P32a
    Train_32a_x = Bar32a1(Train_data)
    Train_P = Train_data[:,[0]]
    Test_32a_x = Bar32a1(Test_data)
    Test_P = Test_data[:,[0]]
    
    Cpxbar32a = LinearRegression()
    Cpxbar32a.fit(Train_32a_x,Train_P)
    Train_32a_pred = Cpxbar32a.predict(Train_32a_x)
    loss_train = criterion(Train_32a_pred,Train_P)
    
    Test_32a_pred = Cpxbar32a.predict(Test_32a_x)
    loss_test = criterion(Test_32a_pred,Test_P)
    
    train_32a_loss = np.concatenate((train_32a_loss,loss_train),1)
    test_32a_loss = np.concatenate((test_32a_loss,loss_test),1)
    
    #Train model of Nimis
    Train_N_x = BarN(Train_data)
    Train_P = Train_data[:,[0]]
    Test_N_x = BarN(Test_data)
    Test_P = Test_data[:,[0]]
    
    CpxbarN = LinearRegression()
    CpxbarN.fit(Train_N_x,Train_P)
    Train_N_pred = CpxbarN.predict(Train_N_x)
    loss_train = criterion(Train_N_pred,Train_P)
    
    Test_N_pred = CpxbarN.predict(Test_N_x)
    loss_test = criterion(Test_N_pred,Test_P)
    
    train_N_loss = np.concatenate((train_N_loss,loss_train),1)
    test_N_loss = np.concatenate((test_N_loss,loss_test),1)
    
    #P32b iteratively calculate with P32dH
    T0_train = np.full((len(Train_T),1),1000.)
    T0_test = np.full((len(Test_T),1),1000.)
    
    for k in range(200):
        if k == 0:
            variable = np.concatenate((T0_train,np.log(T0_train),Bar32b2(Train_data)),1)
            P32b_iter_train = Cpxbar32b.predict(variable)
            T32dH_iter_train = (93100+544*P32b_iter_train)/Cpxtem32dH.predict(Train_32dH_x)
        else:
            variable = np.concatenate((T32dH_iter_train,np.log(T32dH_iter_train),Bar32b2(Train_data)),1)
            P32b_iter_train = Cpxbar32b.predict(variable)
            T32dH_iter_train = (93100+544*P32b_iter_train)/Cpxtem32dH.predict(Train_32dH_x)
            
    for l in range(200):
        if l == 0:
            variable = np.concatenate((T0_test,np.log(T0_test),Bar32b2(Test_data)),1)
            P32b_iter_test = Cpxbar32b.predict(variable)
            T32dH_iter_test = (93100+544*P32b_iter_test)/Cpxtem32dH.predict(Test_32dH_x)
        else:
            variable = np.concatenate((T32dH_iter_test,np.log(T32dH_iter_test),Bar32b2(Test_data)),1)
            P32b_iter_test = Cpxbar32b.predict(variable)
            T32dH_iter_test = (93100+544*P32b_iter_test)/Cpxtem32dH.predict(Test_32dH_x)
    
    loss_train_32b_iter = criterion(P32b_iter_train,Train_P)
    loss_test_32b_iter = criterion(P32b_iter_test,Test_P)
    loss_train_32dH_iter = criterion(T32dH_iter_train,Train_T)
    loss_test_32dH_iter = criterion(T32dH_iter_test,Test_T)
    
    train_32b_loss_iter = np.concatenate((train_32b_loss_iter,loss_train_32b_iter),1)
    test_32b_loss_iter = np.concatenate((test_32b_loss_iter,loss_test_32b_iter),1)
    train_32dH_loss_iter = np.concatenate((train_32dH_loss_iter,loss_train_32dH_iter),1)
    test_32dH_loss_iter = np.concatenate((test_32dH_loss_iter,loss_test_32dH_iter),1)
    
    #P32a iteratively calculate with P32dH
    for m in range(200):
        if m == 0:
            variable = np.concatenate((T0_train,np.log(T0_train),Bar32a2(Train_data)),1)
            P32a_iter_train = Cpxbar32a.predict(variable)
            T32dH_iter_train = (93100+544*P32a_iter_train)/Cpxtem32dH.predict(Train_32dH_x)
        else:
            variable = np.concatenate((T32dH_iter_train,np.log(T32dH_iter_train),Bar32a2(Train_data)),1)
            P32a_iter_train = Cpxbar32a.predict(variable)
            T32dH_iter_train = (93100+544*P32a_iter_train)/Cpxtem32dH.predict(Train_32dH_x)
            
    for n in range(200):
        if n == 0:
            variable = np.concatenate((T0_test,np.log(T0_test),Bar32a2(Test_data)),1)
            P32a_iter_test = Cpxbar32a.predict(variable)
            T32dH_iter_test = (93100+544*P32a_iter_test)/Cpxtem32dH.predict(Test_32dH_x)
        else:
            variable = np.concatenate((T32dH_iter_test,np.log(T32dH_iter_test),Bar32a2(Test_data)),1)
            P32a_iter_test = Cpxbar32a.predict(variable)
            T32dH_iter_test = (93100+544*P32a_iter_test)/Cpxtem32dH.predict(Test_32dH_x)
    
    loss_train_32a_iter = criterion(P32a_iter_train,Train_P)
    loss_test_32a_iter = criterion(P32a_iter_test,Test_P)
    
    train_32a_loss_iter = np.concatenate((train_32a_loss_iter,loss_train_32a_iter),1)
    test_32a_loss_iter = np.concatenate((test_32a_loss_iter,loss_test_32a_iter),1)
    
    #New thermometric model
    Train_NT_x = NewT(Train_data)
    Train_T = Train_data[:,[1]]
    Test_NT_x = NewT(Test_data)
    Test_T = Test_data[:,[1]]
    
    CpxtemN = LinearRegression()
    CpxtemN.fit(Train_NT_x,Train_T)
    Train_NT_pred = CpxtemN.predict(Train_NT_x)
    loss_train = criterion(Train_NT_pred,Train_T)
    
    Test_NT_pred = CpxtemN.predict(Test_NT_x)
    loss_test = criterion(Test_NT_pred,Test_T)
    
    train_NT_loss = np.concatenate((train_NT_loss,loss_train),1)
    test_NT_loss = np.concatenate((test_NT_loss,loss_test),1)
            
#P32b loss histogram
print('P32b_calibratioin_loss:',np.mean(train_32b_loss))
print('P32b_validation_loss:',np.mean(test_32b_loss))
plt.hist(train_32b_loss.tolist(),histtype='stepfilled', color='w', edgecolor='lightcoral')
plt.hist(test_32b_loss.tolist(),histtype='stepfilled', color='w', edgecolor='goldenrod')
plt.xlim(1,4.5,0.5)
plt.ylim(0,25,5)
plt.xlabel('Loss')
plt.ylabel('Frequency')
fig1 = plt.figure(figsize=(4, 3), dpi=300)
plt.show()

#Thermometric models loss histogram
print('P32dH_calibration_loss:',np.mean(train_32dH_loss))
print('P32dH_validation_loss:',np.mean(test_32dH_loss))
print('P32dH_iteration_calibration_loss:',np.mean(train_32dH_loss_iter))
print('P32dH_iteration_validation_loss:',np.mean(test_32dH_loss_iter))
plt.hist(train_32dH_loss.tolist(),histtype='stepfilled', color='w', edgecolor='olivedrab')
plt.hist(test_32dH_loss.tolist(),histtype='stepfilled', color='w', edgecolor='cadetblue')
plt.hist(train_32dH_loss_iter.tolist(),histtype='stepfilled', color='w', edgecolor='lightcoral')
plt.hist(test_32dH_loss_iter.tolist(),histtype='stepfilled', color='w', edgecolor='goldenrod')
plt.hist(train_NT_loss.tolist(),histtype='stepfilled', color='w', edgecolor='pink')
plt.hist(test_NT_loss.tolist(),histtype='stepfilled', color='w', edgecolor='lightblue')
plt.xlabel('Loss')
plt.ylabel('Frequency')
fig2 = plt.figure(figsize=(4, 3), dpi=300)
plt.show()

#P32a iteratively 32b Nimisloss histogram
print('P32a_calibration_loss:',np.mean(train_32a_loss))
print('P32a_validation_loss:',np.mean(test_32a_loss))
print('P32a_iteration_calibration_loss:',np.mean(train_32a_loss_iter))
print('P32a_iteration_validation_loss:',np.mean(test_32a_loss_iter))
print('P32b_calibration_loss:',np.mean(train_32b_loss_iter))
print('P32b_validation_loss:',np.mean(test_32b_loss_iter))
print('Nimis_calibration_loss:',np.mean(train_N_loss))
print('Nimis_validation_loss:',np.mean(test_N_loss))
plt.hist(train_32a_loss.tolist(),histtype='stepfilled', color='w', edgecolor='olivedrab')
plt.hist(test_32a_loss.tolist(),histtype='stepfilled', color='w', edgecolor='cadetblue')
plt.hist(train_32a_loss_iter.tolist(),histtype='stepfilled', color='w', edgecolor='peru')
plt.hist(test_32a_loss_iter.tolist(),histtype='stepfilled', color='w', edgecolor='goldenrod')
plt.hist(train_32b_loss_iter.tolist(),histtype='stepfilled', color='w', edgecolor='pink')
plt.hist(test_32b_loss_iter.tolist(),histtype='stepfilled', color='w', edgecolor='lightblue')
plt.hist(train_N_loss.tolist(),histtype='stepfilled', color='w', edgecolor='black')
plt.hist(test_N_loss.tolist(),histtype='stepfilled', color='w', edgecolor='grey')
plt.xlabel('Loss')
plt.ylabel('Frequency')
fig3 = plt.figure(figsize=(4, 3), dpi=300)
plt.show()

