from sklearn.preprocessing import Normalizer
import torch
import numpy as np
import os
import argparse
from datetime import datetime
import pandas as pd
import random

random.seed(101)

# time
now = datetime.now()
d_print = now.strftime("%d%m_%H%M")

# set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device=', device)

folder='torch_data/'
Save_folder ='work_dir/'

parser = argparse.ArgumentParser(description='GraphLeak')


parser.add_argument('--Pressure', type=bool, default=True, help='Pressure Value?')
parser.add_argument('--Flow', type=bool, default=True, help='Flow Values')
parser.add_argument('--Volume', type=bool, default=True, help='Volume Values')

parser.add_argument('--Noise', type=bool, default=False, help='Do you want to Gaussian noise in their data?')
parser.add_argument('--mu', type=float, default=0.0, help='If you chose Gaussian noise, whats is the mean value?')
parser.add_argument('--sigma', type=float, default=0.1, help='If you chose Gaussian noise, whats is the standard deviation value?')

parser.add_argument('--Nodes_normalization', type=bool, default=True, help='Do you want normalize the values between nodes?')
parser.add_argument('--Data_Normalization', type=bool, default=True, help='Do you want normalize the values on the range 0 to 1?')

parser.add_argument('--WDS_name', type=str, default='WDS1', help='What is the name of WDS?')
parser.add_argument('--data_name', type=str, default='data.csv', help='What is the name of data?')

args = parser.parse_args()

dt_P = args.Pressure
dt_F = args.Flow
dt_V = args.Volume

noise = args.Noise
mu = args.mu
sigma = args.sigma

N_norm = args.Nodes_normalization
D_norm = args.Data_Normalization

WDS_n = args.WDS_name
dataName = args.data_name

absolutepath = os.path.abspath(__file__)
print('absolutepath=',absolutepath) 
fileDirectory = os.path.dirname(absolutepath)
print('fileDirectory=', fileDirectory) 
parentDirectory = os.path.dirname(fileDirectory)
print('parentDirectory=', parentDirectory) 

SaveDirectory = os.path.join(fileDirectory,'WorkDir')
print('SaveDirectory=', SaveDirectory) 



# Load Data

if WDS_n == "WDS1":
    
    nOfn=8 # numero total nós
    nv=5 # numero de simulações de vazamento
    ns = nOfn-nv
    
elif WDS_n == "WDS2":
    
    nOfn=19 # numero total nós
    nv=8 # numero de simulações de vazamento
    ns = nOfn-nv


df = pd.read_csv(dataName,index_col=0)


print('Number of nodes: {}; Number of leak nodes: {}; Number of nodes without leakage: {}'.format(nOfn, nv, ns))

H_flow = []
H_Pressure = []
H_Volume = []
H_Label = []

Label = df.iloc[:,-(nv+4):-4]

for i in range(nOfn):
  H_flow.append('FL'+str(i+2))
  H_Pressure.append('P'+str(i+2))
  H_Volume.append('VL'+str(i+2))

for i in range(nv):
  H_Label.append('L'+str(i+2))


def dataPreprocess(nOfn,nv):
    
    flow = df.iloc[:, 1:nOfn+1]
    pressure = df.iloc[:,nOfn+1 :(nOfn+nOfn)+1]
    volume = df.iloc[:,(nOfn*2)+1 :(nOfn*3)+1]
    
    pressure.columns=H_Pressure
    flow.columns=H_flow
    volume.columns=H_Volume
    
    return pressure, volume, flow

def nodeNorm(pressure, flow, volume):

    for i in range(1,len(H_Pressure)):
        
        pres, fl, vol = pressure, flow, volume
    
        pres[H_Pressure[i]] = pressure[H_Pressure[i]] - pressure[H_Pressure[0]]
        
        fl[H_flow[i]] = flow[H_flow[i]] - flow[H_flow[0]]
        
        vol[H_Volume[i]] = volume[H_Volume[i]] - volume[H_Volume[0]]

    return pres, fl, vol

def noise(data, mu, sigma):
    #d1 = pressure
    d1 = data
    gauss_dimension=np.shape(d1)
     
    noise = np.random.normal(mu, sigma, gauss_dimension)
    
    data_with_noise = d1 + noise

    return data_with_noise

def dataNorm(data):
    from sklearn import preprocessing

    # pressure
    X = preprocessing.minmax_scale(data.values, feature_range=(0, 1))  # Escala entre 0 e 1

    return X

pressure, volume, flow = dataPreprocess(nOfn,nv)

if noise == True:
    pressure = noise(pressure, mu, sigma)
    volume = noise(volume, mu, sigma)
    flow = noise(flow, mu, sigma)
    
if N_norm == True:
    nodeNorm(pressure, flow, volume)
    
if D_norm == True:
    pressure = dataNorm(pressure)
    volume = dataNorm(volume)
    flow = dataNorm(flow)
    
#dict_conc = {'Pressure':0, 'volume':0, 'flow':0}
if dt_P == True:
    d1 = pd.DataFrame(pressure, columns=H_Pressure)
    data = d1
    
if dt_F == True:
    d2 = pd.DataFrame(flow, columns=H_flow)
    data = d2
    
if dt_V == True:
    d3 = pd.DataFrame(volume, columns=H_Volume)
    data = d3

if dt_P == True and dt_F == True and dt_V == True:
    data = pd.concat([d1,d2,d3], ignore_index=False, axis=1)
    
elif dt_P == True and dt_F == True and dt_V == False:
    data = pd.concat([d1,d2], ignore_index=False, axis=1)

elif dt_P == True and dt_F == False and dt_V == True:
    data = pd.concat([d1,d3], ignore_index=False, axis=1)

elif dt_P == False and dt_F == True and dt_V == True:
    data = pd.concat([d2,d3], ignore_index=False, axis=1)


# Label

z=np.shape(pressure)
Labels = np.zeros(z[0])
Label.columns = H_Label

Label['Node'] = pd.DataFrame(Labels)

'''
Label.loc[Label['L2'] == 1,  'Node'] = 1 #
Label.loc[Label['L3'] == 1,  'Node'] = 2 #4
Label.loc[Label['L4'] == 1,  'Node'] = 3 #5
Label.loc[Label['L5'] == 1,  'Node'] = 4 #6
Label.loc[Label['L6'] == 1,  'Node'] = 5 #7

Some models need sequential output.
'''
Label.loc[Label['L2'] == 1,  'Node'] = 1 
Label.loc[Label['L3'] == 1,  'Node'] = 4
Label.loc[Label['L4'] == 1,  'Node'] = 5
Label.loc[Label['L5'] == 1,  'Node'] = 6
Label.loc[Label['L6'] == 1,  'Node'] = 7

y=Label['Node']

print('Label_Nunique:', np.unique(y, return_counts=True))

data.to_csv('data1.csv')#, encoding='utf-8', index=False)

y.to_csv('label1.csv')#, encoding='utf-8', index=False)


