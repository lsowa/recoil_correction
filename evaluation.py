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
import numpy as np
import matplotlib

from helpers import *
from sklearn.preprocessing import StandardScaler
from os.path import exists
from argparse import ArgumentParser

# +
parser = ArgumentParser()
parser.add_argument("-c", "--cuda", dest="cuda", default=0, type=int, help='Cuda number')

parser.add_argument("-f", '--flows', dest="flows", default=8, type=int, help='Number of flows')
parser.add_argument("--nn-hidden", dest="nn_hidden", default=3, type=int, help='Number of hidden layers for the NN')
parser.add_argument("--nn-nodes", dest="nn_nodes", default=200, type=int, help='Number of hidden nodes for NN')
parser.add_argument("--model", dest="model", default='8flows_3layer_200nodes_50000batch/model.pt', type=str, help='Path to model')
parser.add_argument("--output", dest="output", default='eval_dummy/', type=str, help='Path for output files')

parser.add_argument("--test", dest="test", default=1, type=int, help='Test run {True, False}')

args = parser.parse_args("")

print(args)
# -

ensure_dir(args.output)

if exists('/ceph/lsowa/recoil/dt.root'):
    dfdata = load_from_root('/ceph/lsowa/recoil/dt.root', test=args.test)
    dfmc = load_from_root('/ceph/lsowa/recoil/mc.root', test=args.test)
else:
    # when running on cluster
    dfdata = load_from_root('recoil/dt.root', test=args.test)
    dfmc = load_from_root('recoil/mc.root', test=args.test)

pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))

cond = ['metphi','pt_vis_c', 'phi_vis_c','pt_1', 'pt_2','dxy_1', 'dxy_2','dz_1',
        'dz_2','eta_1', 'eta_2','mass_1', 'mass_2','metSumEt']
names = ['uP1_uncorrected', 'uP2_uncorrected']

data = dfdata[names].to_numpy().astype(float)
mc = dfmc[names].to_numpy().astype(float)
cdata = dfdata[cond].to_numpy().astype(float)
cmc = dfmc[cond].to_numpy().astype(float)

# +
# # +
# Z standardize inputs
# # +
input_scaler = StandardScaler()
data = input_scaler.fit_transform(data)
mc = input_scaler.transform(mc)

cond_scaler = StandardScaler()
cdata = cond_scaler.fit_transform(cdata)
cmc = cond_scaler.transform(cmc)
# -

data, mc, cdata, cmc = torch.tensor(data), torch.tensor(mc), torch.tensor(cdata), torch.tensor(cmc)


# +
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
# -

model.load_state_dict(torch.load(args.model))

model.cpu()
z = evaluate_sequential(model, data.float(), cond=cdata.float())
gaussian = pz.sample((100000, ))
density_2d(z.cpu().detach().numpy(), gaussian.cpu().detach().numpy(), 
            dist_label=r'target gaussian $z$', data_label=r'model(y, $\mathrm{cond}_\mathrm{Data}$)= $\hat{z}$', 
            xlim = [-3, 3], ylim = [-3, 3], save_as=args.output+'2d_gaussian_data.pdf')

z = evaluate_sequential(model, mc.float(), cond=cmc.float())
gaussian = pz.sample((100000, ))
density_2d(z.cpu().detach().numpy(), gaussian.cpu().detach().numpy(), 
            dist_label=r'target gaussian $z$', data_label=r'model(y, $\mathrm{cond}_\mathrm{Data}$)= $\hat{z}$', 
            xlim = [-3, 3], ylim = [-3, 3], save_as=args.output+'2d_gaussian_mc.pdf')

# +

# Predict y

z = pz.sample((cmc.shape[0], ))
u = evaluate_sequential(model, z, cmc.float(), rev=True)
u = u.cpu().detach().numpy()

u = input_scaler.inverse_transform(u)
data = input_scaler.inverse_transform(data)
mc = input_scaler.inverse_transform(mc)

# -

density_2d(u, data, 
            dist_label=r'$u^\mathrm{Data}_\parallel$', data_label=r'model(z, $\mathrm{cond}_\mathrm{MC}$)=$\hat{u_\parallel}$', 
            xlim = [-160, 100], ylim = [-80, 30], save_as=args.output+'2d_model_data.pdf')

density_2d(u, mc, 
            dist_label=r'$u^\mathrm{MC}_\parallel$', data_label=r'model(z, $\mathrm{cond}_\mathrm{MC}$)=$\hat{u_\parallel}$', 
            xlim = [-160, 100], ylim = [-80, 30], save_as=args.output+'2d_model_mc.pdf')

# # ### Compare MC -> Data

# u parallel
interval = [-170, 100]
_ = plt.hist(u[:,0], density=True, bins=100, range=interval, label=r'model(z,$c^\mathrm{MC}$)=$u_\parallel$')
_ = plt.hist(data[:,0], histtype=u'step', density=True, bins=100, 
                range=interval, linewidth=2, color='black', label=r'$u^\mathrm{Data}_\parallel$')
_ = plt.hist(dfmc['uP1_uncorrected'].values, histtype=u'step', density=True, 
                bins=100, range=interval, linewidth=2, color='red', label=r'$u^\mathrm{MC}_\parallel$ uncorrected')
plt.xlabel(r'$u_\parallel$')
plt.ylabel('a. u.')
plt.legend()
plt.savefig(args.output+'u_parallel.pdf')
plt.clf()



# u perp
interval = [-80, 80]
_ = plt.hist(u[:,1], density=True, bins=100, range=interval, label=r'model(z,$c^\mathrm{MC}$)=$u_\perp $')
_ = plt.hist(data[:,1], histtype=u'step', density=True, bins=100, range=interval, 
                linewidth=2, color='black', label=r'$u^\mathrm{Data}_\perp $')
_ = plt.hist(dfmc['uP2_uncorrected'].values, histtype=u'step', density=True, bins=100, 
                range=interval, linewidth=2, color='red', label=r'$u^\mathrm{MC}_\perp $ uncorrected')
plt.xlabel(r'$u_\perp$')
plt.ylabel('a. u.')
plt.legend()
plt.savefig(args.output+'u_perp.pdf')
plt.clf()



# +
# response

up = u[:,0]
ptz = dfmc['pt_vis_c']

xmin = 25
xmax = 200

keep = np.logical_and(xmax > ptz, ptz > xmin)
ptz = ptz[keep]
up = up[keep]
r = -up/ptz

bins = np.linspace(xmin,xmax,10)
bin_mids = (bins[1:]-bins[:-1])/2 + bins[:-1]
hist_raw, edges = np.histogram(ptz, bins = bins, range=[xmin, xmax])
hist_weighted, edges = np.histogram(ptz, bins = bins, weights=r, range=[xmin, xmax])

#plt.plot(bin_mids, hist_weighted/hist_raw, '.')
plt.hlines([1], bins.min(), bins.max(), color='black')
plt.errorbar(x=bin_mids, y=hist_weighted/hist_raw,
            xerr=(bins[1:]-bins[:-1])/2, fmt='o', capsize=2)
plt.xlabel(r'$p_\mathrm{T}^Z$ in GeV')
plt.ylabel(r'$\langle \frac{\mathrm{u}_\parallel}{p_\mathrm{T}^Z}\rangle$')
plt.xlim([bins.min(), bins.max()])
plt.savefig(args.output+'response.pdf')
plt.clf()
# -

n, bins, edges = plt.hist(up, histtype=r'step', bins=200, range=[-200, 200], 
                            density=True, label=r'$\mathrm{u}_\parallel$')
n1, bins1, edges1 = plt.hist(ptz, histtype=r'step', bins=200, range=[-200, 200], 
                                density=True, label=r'$p_\mathrm{T}^Z}\rangle$')
plt.xlim([-200, 200])
plt.legend()
plt.xlabel('GeV')
plt.ylabel('a.u.')
plt.savefig(args.output+'uperp_ptw.pdf')
plt.clf()


csingle_event = torch.concat([cmc[100,:].unsqueeze(0)]*10000)

# +
z = pz.sample((csingle_event.shape[0], ))
z = evaluate_sequential(model, z, cond=csingle_event.float(), rev=True)

u = input_scaler.inverse_transform(u)
# -

density_2d(u, mc, 
            dist_label=r'target $y_\mathrm{MC}$', data_label=r'model(z, $\mathrm{cond}^\mathrm{fixed}_\mathrm{MC}$)=$\hat{y}$', 
            xlim = [-160, 100], ylim = [-80, 30], save_as=args.output+'fixedevent.pdf')
