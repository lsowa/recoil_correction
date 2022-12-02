import uproot
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from os.path import exists


cond = ['metphi','pt_vis_c', 'phi_vis_c','pt_1', 'pt_2','dxy_1', 'dxy_2','dz_1',
        'dz_2','eta_1', 'eta_2','mass_1', 'mass_2','metSumEt']
target_names = ['uP1_uncorrected', 'uP2_uncorrected']


def load_from_root(path, test=False):
    print('Reading files from: ', path)
    df = []
    for file in uproot.iterate(path+":ntuple", library="pd"):
        df.append(pd.DataFrame.from_dict(file))
        print('No. ', len(df))
        if test and len(df)>=1: break
    df = pd.concat(df)     
    return df

def seperate_cond(df):
    data = df[target_names].to_numpy().astype(float)
    conditions = df[cond].to_numpy().astype(float)
    return data, conditions

def standardize(data, mc):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    mc = scaler.transform(mc)
    return data, mc, scaler


class DataManager(object):
    def __init__(self, test=0):
        self.test = test
        self.dfdata = self.load_from_root('dt.root')
        self.dfmc = self.load_from_root('mc.root')
    
        self.separate_cond()
        self.standardize()
        self.train_test_split()
        self.to_tensor()
        
    def load_from_root(self, file):
        if exists('/ceph/lsowa/recoil/dt.root'):
            return load_from_root('/ceph/lsowa/recoil/'+file, test=self.test)
        else: # when running on cluster
            return load_from_root('recoil/'+file, test=self.test)
    
    def separate_cond(self):
        self.data, self.cdata = seperate_cond(self.dfdata)
        self.mc, self.cmc = seperate_cond(self.dfmc)

    def standardize(self):
        self.data, self.mc, self.input_scaler = standardize(self.data, self.mc)
        self.cdata, self.cmc, self.cond_scaler = standardize(self.cdata, self.cmc)
    
    def train_test_split(self, test_size=0.2):
        self.data, self.data_val, self.cdata, self.cdata_val = train_test_split(self.data, self.cdata, test_size=test_size)
    
    def to_tensor(self):
        self.data = torch.tensor(self.data)
        self.mc = torch.tensor(self.mc)
        self.cdata = torch.tensor(self.cdata)
        self.cmc = torch.tensor(self.cmc)
        self.data_val = torch.tensor(self.data_val)
        self.cdata_val = torch.tensor(self.cdata_val)

        print('Train (Data): ', list(self.data.shape), ' Conditions: ', list(self.cdata.shape))
        print('Val (Data): ', list(self.data_val.shape), ' Conditions: ', list(self.cdata_val.shape))
        print('Test (MC): ', list(self.mc.shape), ' Conditions: ', list(self.cmc.shape))
