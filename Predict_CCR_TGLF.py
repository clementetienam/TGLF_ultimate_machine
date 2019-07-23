# -*- coding: utf-8 -*-
"""
@author: Dr Clement Etienam
Supervisor: Professor Kody Law
In this script we will predict on unseen data for TGLF using the pre-trained CCR model
The test data must be in CSV format with the corresponding column names shown below

Input columns (24 columns) arrangement (1-24 in descending order):
RMIN_LOC
Q_LOC
XNUE
VPAR_1
RMAJ_LOC
DRMAJDX_LOC
VEXB_SHEAR
TAUS_2
DEBYE
ZEFF
KAPPA_LOC
BETAE
S_KAPPA_LOC
RLTS_2
RLTS_1
VPAR_SHEAR_1
RLNS_1
Q_PRIME_LOC
RLNS_2
RLNS_3
P_PRIME_LOC
DELTA_LOC
AS_3
AS_2

Output column arrangement (1-6 in descending order):  
    
OUT_tur_STRESS_TOR_i
OUT_tur_ENERGY_FLUX_i
OUT_tur_PARTICLE_FLUX_2
OUT_tur_PARTICLE_FLUX_1
OUT_tur_ENERGY_FLUX_1
OUT_tur_PARTICLE_FLUX_3
"""
from __future__ import print_function
print(__doc__)
##-------------------------Import Libraries-----------------------------------------------------------

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.stats import rankdata, norm
from scipy import interpolate
import pandas as pd
print('CCR for prediction on unseen data')
print('')
print('--------------------DEFINE FUNCTIONS----------------------------------')

def interpolatebetween(xtrain,cdftrain,xnew):
    numrows1=len(xnew)
    numcols = len(xnew[0])
    norm_cdftest2=np.zeros((numrows1,numcols))
    for i in range(numcols):
        f = interpolate.interp1d((xtrain[:,i]), cdftrain[:,i],kind='nearest')
        cdftest = f(xnew[:,i])
        norm_cdftest2[:,i]=np.ravel(cdftest)
    return norm_cdftest2

def gaussianizeit(input1):
    numrows1=len(input1)
    numcols = len(input1[0])
    newbig=np.zeros((numrows1,numcols))
    for i in range(numcols):
        input11=input1[:,i]
        newX = norm.ppf(rankdata(input11)/(len(input11) + 1))
        newbig[:,i]=newX.T
    return newbig
print('')
print('-------------------DEFINE INPUT COLUMN NAMES--------------------------')

#------------------Begin Code-----------------------------------------------------------------#
print('Load the input data you want to predict from')
data1=pd.read_csv("TGLFupdated.csv")
data1.drop(data1.columns[[2,8,18,19,24,26]], axis=1, inplace=True)
ruthlist=list(data1)
print('Input columns must be arranged in this format')
print('')
for col in data1.columns: 
    print(col)

print('')
print('outputs are arranged in this format')
print('')
print(' 1st column=OUT_tur_STRESS_TOR_i' )
print(' 2nd column=OUT_tur_ENERGY_FLUX_i' )
print(' 3rd column=OUT_tur_PARTICLE_FLUX_2' )
print(' 4th column=OUT_tur_PARTICLE_FLUX_1' )
print(' 5th column=OUT_tur_ENERGY_FLUX_1' )
print(' 6th column=OUT_tur_PARTICLE_FLUX_3' )
print('')
print('-----------------USE ORIGINAL TRAINING DATA TO RECOVER DISTRIBUTIONS--')

##---------------------Begin Program-------------------------------##
print('get original distribution of training data')
df=pd.read_csv("TGLFupdated.csv")
data=df.values 
test=data
output=data[:,[2,8,18,19,24,26]]
testdata=np.delete(test, [2,8,18,19,24,26], axis=1)
inputfirst=testdata #original training data
input2=testdata #transformed training data
print('Convert the trainig data to Gausian and standardise')
scaler = MinMaxScaler()
input2=gaussianizeit(input2)
input2= scaler.fit(input2).transform(input2)
print('')
print('Standardize and normalize (make gaussian) the test data')
print('-------------------INPUT TEST DATA IN CSV FORMAT----------------------')
filename=input('Enter the name of the test input data in csv format: ')
data4=pd.read_csv(filename)
data22 = data4[ruthlist]
data22=data22.values
inputsecond=data22 #Assume we have test data
clement=interpolatebetween(inputfirst,input2,inputsecond)
##---------------------Begin Program-------------------------------------##
numrows=len(clement)    # rows of input
numrowstest=numrows
numcols = len(clement[0])
numruth = numcols
inputtest=clement

print('----------------START PREDICTION ON TEST DATA-------------------------')
outputtrain=np.reshape((output),(-1,6),'F') #select the first 300000 for training
ydamir=outputtrain
outputtrain1=outputtrain
scaler1 = MinMaxScaler()
(scaler1.fit(ydamir))
ydamir=(scaler1.transform(ydamir))

outputtest1=np.reshape((output),(-1,6),'F')

filename1= 'CCR_TGLF_MACHINE.asv'
loaded_model = pickle.load(open(filename1, 'rb'))
clementanswer= loaded_model.predict(inputtest)
clementanswer=scaler1.inverse_transform(clementanswer)

print('')
print('----------------END OF PREDICTION-------------------------------------')

print('')
print('--------------------SAVE PREDCTION TO FILE----------------------------')
header = '%16s\t%16s\t%16s\t%16s\t%16s\t%16s'%('STRESS_TOR_i','ENERGY_FLUX_i','PARTICLE_FLUX_2','PARTICLE_FLUX_1','ENERGY_FLUX_1','PARTICLE_FLUX_3')
valueCCR2=clementanswer
np.savetxt('CCR_TGLF_prediction.out',valueCCR2[0:20000,:], fmt = '%4.16f',delimiter='\t',header=header, newline = '\n')
print('')
print('----------------------------------------------------------------------')
print('PROGRAM EXECUTED SUCCESFULY')
