#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

def import_Excel_data(file_name,sheet_name):
    Cpx_data = pd.read_excel(file_name,sheet_name=sheet_name,usecols="AE,AF,AH,AI:AM,AP",skiprows=1)#Si,Ti,Cr,Fe,Mn,Mg,Ca,Na,AlVI
    Cpx_data = Cpx_data.fillna(0)
    Cpx_data.columns = [c.replace('.1','') for c in Cpx_data.columns]
    
    P_data = pd.read_excel(file_name,sheet_name=sheet_name,usecols="E",skiprows=1)
    P_data.columns = [c.replace('.1','') for c in P_data.columns]
    
    return Cpx_data,P_data

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu') 

#Import data
Train_data,Train_data_P = import_Excel_data('Table S1.xlsx','calibration dataset')
Test_data,Test_data_P = import_Excel_data('Table S1.xlsx','test dataset')
Train_data = torch.tensor(Train_data.values,dtype = torch.float32).to(device)
Train_data_P = torch.tensor(Train_data_P.values,dtype = torch.float32).to(device)
Test_data = torch.tensor(Test_data.values,dtype = torch.float32).to(device)
Test_data_P = torch.tensor(Test_data_P.values,dtype = torch.float32).to(device)

class baromodel(nn.Module):
    def __init__(self):
        super(baromodel,self).__init__()
        self.input_Layer1 = nn.Linear(5,1)
        self.input_Layer2 = nn.Linear(5,1,bias=False)
        self.a = torch.nn.Parameter(torch.tensor([1.],requires_grad=True))#parameter of Al_VI in NCT
        self.b = torch.nn.Parameter(torch.tensor([1.],requires_grad=True))#parameter before NCT

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
        NCT = ((self.a*AlVI))/(self.input_Layer2(X_2)+self.a*AlVI)

        P_pred = self.b*NCT*torch.log(AlVI) + Linear
        return P_pred
    
#Hyperparameter
cpx_barometer = baromodel().to(device)
criterion = nn.MSELoss().to(device)
lr = 0.01
optimizer = torch.optim.Adam(cpx_barometer.parameters(),lr=lr)

for it in range(1000):
    y_train_pred = cpx_barometer(Train_data)
    loss_train = criterion(y_train_pred,Train_data_P)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    if it%5000 == 0:
        print(it,"loss_train:",loss_train.item())

y_test_pred = cpx_barometer(Test_data)
loss_test = criterion(y_test_pred,Test_data_P)

print("loss_train:",loss_train.item())
print("loss_test:",loss_test.item())
plt.scatter(Train_data_P.cpu().detach().numpy(),y_train_pred.cpu().detach().numpy())
plt.scatter(Test_data_P.cpu().detach().numpy(),y_test_pred.cpu().detach().numpy())

for parameters in cpx_barometer.parameters():
    print(parameters)






