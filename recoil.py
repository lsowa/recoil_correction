# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.6.9 ('cluster')
#     language: python
#     name: python3
# ---

# +
import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import pandas as pd
import torch.distributions as dist
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import matplotlib

from os.path import exists
from helpers import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser

matplotlib.rcParams['figure.figsize'] = [15, 10]
plt.rcParams.update({'axes.labelsize': 14})
# -

parser = ArgumentParser()
parser.add_argument("-c", "--cuda", dest="cuda", default=0, type=int, help='Cuda number')
parser.add_argument("--batch", dest="batch", default=5000, type=int, help='batch size')
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help='learning rate')
parser.add_argument("--test", dest="test", default=0, type=int, help='Test run {True, False}')

parser.add_argument("-f", '--flows', dest="flows", default=2, type=int, help='Number of flows')
parser.add_argument("--nn-hidden", dest="nn_hidden", default=2, type=int, help='Number of hidden layers for the NN')
parser.add_argument("--nn-nodes", dest="nn_nodes", default=150, type=int, help='Number of hidden nodes for NN')

args = parser.parse_args()

folder = 'output/'
ensure_dir(folder)
device = torch.device('cuda:'+str(args.cuda))

if exists('/ceph/lsowa/recoil/dt.root'):
    dfdata = load_from_root('/ceph/lsowa/recoil/dt.root', test=args.test)
    dfmc = load_from_root('/ceph/lsowa/recoil/mc.root', test=args.test)
else:
    # when running on cluster
    dfdata = load_from_root('recoil/dt.root', test=args.test)
    dfmc = load_from_root('recoil/mc.root', test=args.test)

print('MC shape ', dfmc.shape)
print('Data shape ', dfdata.shape)

cond = ['pt_vis_c', 'phi_vis_c', 'dxyErr_1', 'dxyErr_2', 'dxy_1', 'dxy_2',  'dxybs_1',
        'dxybs_2', 'dzErr_1', 'dzErr_2', 'dz_1', 'dz_2', 'eta_1', 'eta_2', 'highPtId_1',
        'highPtId_2', 'highPurity_1', 'highPurity_2', 'mass_1', 'mass_2', 'metSumEt', 'metcov00',
        'metcov01', 'metcov10', 'metcov11', 'metphi', 'mjj', 'phi_1', 'phi_2', 'ptErr_1', 
        'ptErr_2', 'pt_1', 'pt_2']
cond = ['metphi','pt_vis_c', 'phi_vis_c','pt_1', 'pt_2','dxy_1', 'dxy_2','dz_1',
        'dz_2','eta_1', 'eta_2','mass_1', 'mass_2','metSumEt']
names = ['uP1_uncorrected', 'uP2_uncorrected']


data = dfdata[names].to_numpy().astype(float)
mc = dfmc[names].to_numpy().astype(float)
cdata = dfdata[cond].to_numpy().astype(float)
cmc = dfmc[cond].to_numpy().astype(float)

ptz = dfmc['pt_vis_c']

if False:#args.test:
    n = 2000
    data = data[:n,:]
    cdata = cdata[:n,:]
    #mc = mc[:n,:]
    #cmc = cmc[:n,:]
    #ptz = ptz[:n] # MC


# +
# Z standardize inputs
# +
input_scaler = StandardScaler()
data = input_scaler.fit_transform(data)
mc = input_scaler.transform(mc)

cond_scaler = StandardScaler()
cdata = cond_scaler.fit_transform(cdata)
cmc = cond_scaler.transform(cmc)
# -

data, data_val, cdata, cdata_val = train_test_split(data, cdata, test_size=0.2)

data, mc, cdata, cmc = torch.tensor(data), torch.tensor(mc), torch.tensor(cdata), torch.tensor(cmc)
data_val, cdata_val = torch.tensor(data_val).to(device), torch.tensor(cdata_val).to(device)

print('Train (Data): ', data.shape, ' Conditions: ', cdata.shape)
print('Val (Data): ', data_val.shape, ' Conditions: ', cdata_val.shape)
print('Test (MC): ', mc.shape, ' Conditions: ', cmc.shape)


# Setup Model

def mlp_constructor(input_dim=2, out_dim=2, hidden_nodes=args.nn_nodes):
    
    layers = [nn.Linear(input_dim, hidden_nodes), nn.ReLU()]
    for n in range(args.nn_hidden-1):
        layers.append(nn.Linear(hidden_nodes, hidden_nodes))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_nodes, out_dim))
    
    model = nn.Sequential(*layers)
    return model

model = Ff.SequenceINN(2)
for k in range(args.flows):
    model.append(Fm.RNVPCouplingBlock, subnet_constructor=mlp_constructor, 
                    clamp=2, cond=0, cond_shape=(cmc.shape[1],))

#
# Training
#

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, 
                                                    patience=3, verbose=True)

cdata.shape

dataset = TensorDataset(data, cdata)
loader = DataLoader(dataset, shuffle=True, batch_size=args.batch, num_workers=4)

# +
pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
model.to(device)

def nll(z, log_jac):
    zz = torch.sum(z**2, dim=-1)
    neg_log_likeli = 0.5 * zz - log_jac
    loss = torch.mean(neg_log_likeli)
    return loss

losses = []
losses_val = []
nbatches = len(loader)
stopper=0
best_loss=np.inf
epoch = 0
best_model_dict = None
while stopper<=15:
    epo_loss = 0
    for d, c in loader:
        d = d.to(device)
        c = c.to(device)
        
        z, log_jac = model(d.float(), c=[c.float()])
        loss = nll(z, log_jac)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epo_loss += loss.cpu().detach().numpy()
    epo_loss /= nbatches
    epo_loss = round(epo_loss,5)
    scheduler.step(epo_loss)
    losses.append(epo_loss)
    
    # validation
    z, log_jac = model(data_val.float(), c=[cdata_val.float()])
    loss_val = nll(z, log_jac)
    loss_val = loss_val.cpu().detach().numpy()
    losses_val.append(loss_val)
    if loss_val >= best_loss: 
        stopper+=1
    else:
        best_loss = loss_val
        stopper = 0
        best_model_dict = model.state_dict()
    epoch += 1
    print('Epoch: {:.0f}; train Loss: {:.3f}; val Loss: {:.3f}; stopper: {:.0f}; best val loss: {:.3f}\n'.format(epoch, epo_loss, loss_val, stopper, best_loss))
    
    if epoch >= args.test and args.test > 0:
        break


# -
plt.plot(losses, label='training')
plt.plot(losses_val, label='validation')
plt.vlines(len(losses)-stopper, ymin=np.min(losses), ymax=np.max(losses), 
            label='best model', colors=['grey'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(folder+'loss.pdf')

model.load_state_dict(best_model_dict)
torch.save(model.state_dict(), folder+'model.pt')
os.system('cp ' + os.path.basename(__file__) + ' ' + folder + 'code.py') 

print('Training done')
