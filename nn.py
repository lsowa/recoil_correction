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
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
import matplotlib

from torch import nn
from helpers import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from os.path import exists
from argparse import ArgumentParser

matplotlib.use('Agg')

# +

parser = ArgumentParser()

parser.add_argument("-c", "--cuda", dest="cuda", default=0, type=int, help='Cuda number')
parser.add_argument("--nn-hidden", dest="nn_hidden", default=3, type=int, help='Number of hidden layers for the NN')
parser.add_argument("--nn-nodes", dest="nn_nodes", default=200, type=int, help='Number of hidden nodes for NN')
parser.add_argument("--lr", dest="lr", default=0.01, type=float, help='Learning rate (to start with)')
parser.add_argument("--batch", dest="batch", default=500, type=int, help='Batch Size')
parser.add_argument("--output", dest="output", default='nn_dummy/', type=str, help='Path for output files')
parser.add_argument("--test", dest="test", default=0, type=int, help='Test run {True, False}')
args = parser.parse_args()
# pass default arguments if executed as ipynb
try: 
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell': args = parser.parse_args("") 
except:
    args = parser.parse_args()
# -

ensure_dir(args.output)
device = args.cuda

# +
#
# Load data
#
# -

if exists('/ceph/lsowa/recoil/dt.root'):
    # when on local machine
    dfdata = load_from_root('/ceph/lsowa/recoil/dt.root', test=args.test)
    dfmc = load_from_root('/ceph/lsowa/recoil/mc.root', test=args.test)
else:
    # when running on cluster
    dfdata = load_from_root('recoil/dt.root', test=args.test)
    dfmc = load_from_root('recoil/mc.root', test=args.test)

cond = ['metphi','pt_vis_c', 'phi_vis_c','pt_1', 'pt_2','dxy_1', 'dxy_2','dz_1',
        'dz_2','eta_1', 'eta_2','mass_1', 'mass_2','metSumEt']
names = ['uP1_uncorrected', 'uP2_uncorrected']

data = dfdata[names].to_numpy().astype(float)
mc = dfmc[names].to_numpy().astype(float)
cdata = dfdata[cond].to_numpy().astype(float)
cmc = dfmc[cond].to_numpy().astype(float)

del dfdata, dfmc

# +
#
# Preprocess data
#

# +
# Z standardize inputs
input_scaler = StandardScaler()
data = input_scaler.fit_transform(data)
mc = input_scaler.transform(mc)

cond_scaler = StandardScaler()
cdata = cond_scaler.fit_transform(cdata)
cmc = cond_scaler.transform(cmc)

# +
# train test split
data, data_val, cdata, cdata_val = train_test_split(data, cdata, test_size=0.2)

data, mc, cdata, cmc = torch.tensor(data), torch.tensor(mc), torch.tensor(cdata), torch.tensor(cmc)
data_val, cdata_val = torch.tensor(data_val).to(device), torch.tensor(cdata_val).to(device)
# -

data, mc, cdata, cmc = torch.tensor(data), torch.tensor(mc), torch.tensor(cdata), torch.tensor(cmc)

print('Train (Data): ', data.shape, ' Conditions: ', cdata.shape)
print('Val (Data): ', data_val.shape, ' Conditions: ', cdata_val.shape)
print('Test (MC): ', mc.shape, ' Conditions: ', cmc.shape)


# +
#
# Model Setup
#

# +


model = Mlp(input_neurons=cdata.shape[1], hidden_neurons=args.nn_nodes, output_neurons=2, hiddenlayers=args.nn_hidden)

# +
#
# model training
#
# -

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, 
                                                    patience=3, verbose=True)
mse = nn.MSELoss()

#data, cdata = data[:100,:].to(device), cdata[:100,:].to(device)
dataset = TensorDataset(data, cdata)
loader = DataLoader(dataset, shuffle=True, batch_size=args.batch, num_workers=4, pin_memory=True)



# +
pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
model.to(device)

losses = []
losses_val = []
nbatches = len(loader)
stopper=0
best_loss=np.inf
epoch = 0
loss_val = np.inf
best_model_dict = None

while stopper<=15:
    epo_loss = 0
    for d, c in loader:
        d = d.to(device)
        c = c.to(device)
        
        u = model(c.float())
        loss = mse(u.float(), d.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epo_loss += loss.cpu().detach().numpy()
    epo_loss /= nbatches
    epo_loss = round(epo_loss,5)
    scheduler.step(epo_loss)
    losses.append(epo_loss)
    
    # validation
    u = model(cdata_val.float())
    #z, log_jac = evaluate_sequential(model, data_val.float(), cond=cdata_val.float())
    loss_val = mse(u.float(), data_val.float())
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
plt.savefig(args.output+'loss.pdf')
plt.clf()

model.load_state_dict(best_model_dict)

torch.save(model.state_dict(), args.output+'model.pt')
#os.system('cp ' + os.path.basename(__file__) + ' ' + args.output + 'code.py') 


