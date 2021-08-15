#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import math

def import_Excel_data(file_name,sheet_name):
    Cpx_data = pd.read_excel(file_name,sheet_name=sheet_name,usecols="AE,AF,AH,AI:AM,AP",skiprows=1)#Si,Ti,Cr,Fe,Mn,Mg,Ca,Na,AlVI
    Cpx_data = Cpx_data.fillna(0)
    Cpx_data.columns = [c.replace('.1','') for c in Cpx_data.columns]
    
    P_data = pd.read_excel(file_name,sheet_name=sheet_name,usecols="E",skiprows=1)
    P_data.columns = [c.replace('.1','') for c in P_data.columns]
    
    return Cpx_data,P_data

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')#utilize GPU for computation 

Data_x,P = import_Excel_data('Table S1.xlsx','Cali&Validation data')#load train and test data (i.e. calibration and validation data in text)
Data_x = np.array(Data_x.values)
P = np.array(P.values)
Data = np.concatenate((Data_x,P),axis=1)

train_loss = torch.tensor([]).to(device)#tensor used to record loss value of each loop
test_loss = torch.tensor([]).to(device)

#built the non-linear barometric model
class baromodel(nn.Module):
    def __init__(self):
        super(baromodel,self).__init__()
        self.input_Layer1 = nn.Linear(5,1)
        self.input_Layer2 = nn.Linear(5,1,bias=False)
        self.a = torch.nn.Parameter(torch.tensor([1.],requires_grad=True))
        self.b = torch.nn.Parameter(torch.tensor([1.],requires_grad=True))

    def forward(self,X):
        Si = X[:,[0]]
        Ti = X[:,[1]]
        Cr = X[:,[2]]
        Fe = X[:,[3]]
        Mn = X[:,[4]]
        Mg = X[:,[5]]
        Ca = X[:,[6]]
        Na = X[:,[7]]
        AlVI = X[:,[8]]
        
        X_1 = torch.cat((Si,Fe,Mg,Ca,Na),1)
        Linear = self.input_Layer1(X_1)
        
        X_2 = torch.cat((Ti,Cr,Fe,Mn,Mg),1)
        NLT = ((self.a*AlVI))/(self.input_Layer2(X_2)+self.a*AlVI)*torch.log(AlVI)

        P_pred = self.b*NLT + Linear
        return P_pred


for j in range(100):
    np.random.seed(j)

    for i in range (7):
        if i==0:
            data = Data[Data[:,9] == 0.001,:]#extract data of P=0.001kbar
            np.random.shuffle(data)#randomly sort
            a = math.floor(0.8*len(data))
            b = math.floor(len(data))
            Test_data = data[a:b]#extract the first 80% as calibration dataset
            Train_data = data[0:a]#the remaining data as validation dataset
        else:
            idx = np.where((Data[:,9]>(2*i-2+0.00103))&(Data[:,9]<=2*i))#extract data of 2i-2<P<=2i,except 0.001kbar data
            data = Data[idx]
            np.random.shuffle(data)#randomly sort
            a = math.floor(0.8*len(data))
            b = math.floor(len(data))
            Train_data = np.concatenate((Train_data,data[0:a,:]),0)
            Test_data = np.concatenate((Test_data,data[a:b,:]),0)

    Train_data_x = torch.tensor(Train_data[:,0:9],dtype=torch.float32).to(device)#load data into GPU
    Train_data_P = torch.tensor(Train_data[:,[9]],dtype=torch.float32).to(device)
    Test_data_x = torch.tensor(Test_data[:,0:9],dtype=torch.float32).to(device)
    Test_data_P = torch.tensor(Test_data[:,[9]],dtype=torch.float32).to(device)
    cpx_barometer = baromodel().to(device)#load model into GPU
    #cpx_barometer.load_state_dict(torch.load("cpxbar_para.pth"))#load pretrained parameters
    criterion = nn.MSELoss()#define loss function
    lr = 0.01#set initial learning rate
    optimizer = torch.optim.Adam(cpx_barometer.parameters(),lr=lr)
    #train barometric model
    for it in range(20000):
        y_train_pred = cpx_barometer(Train_data_x)
        loss_train = criterion(y_train_pred,Train_data_P)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        y_test_pred = cpx_barometer(Test_data_x)
        loss_test = criterion(y_test_pred,Test_data_P)
    #save loss values of calibration and validation data
    train_loss = torch.cat((train_loss,torch.tensor([[loss_train]]).to(device)),1)
    test_loss = torch.cat((test_loss,torch.tensor([[loss_test]]).to(device)),1)

#torch.save(cpx_barometer.state_dict(),'cpxbar_para.pth')#save pretrained parameters
plt.hist(train_loss.cpu().detach().numpy().tolist(),histtype='stepfilled', color='w', edgecolor='olivedrab')#plot hisotgram
plt.hist(test_loss.cpu().detach().numpy().tolist(),histtype='stepfilled', color='w', edgecolor='cadetblue')
plt.xlabel('Loss')
plt.ylabel('Frequency')

