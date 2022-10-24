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

matplotlib.use('Agg')

# +
parser = ArgumentParser()
parser.add_argument("-c", "--cuda", dest="cuda", default=0, type=int, help='Cuda number')

parser.add_argument("-f", '--flows', dest="flows", default=8, type=int, help='Number of flows')
parser.add_argument("--nn-hidden", dest="nn_hidden", default=3, type=int, help='Number of hidden layers for the NN')
parser.add_argument("--nn-nodes", dest="nn_nodes", default=200, type=int, help='Number of hidden nodes for NN')
parser.add_argument("--model", dest="model", default='8flows_3layer_200nodes_50000batch/model.pt', type=str, help='Path to model')
parser.add_argument("--output", dest="output", default='eval_dummy/', type=str, help='Path for output files')

parser.add_argument("--test", dest="test", default=1, type=int, help='Test run {True, False}')

# +
# pass default arguments if executed as ipynb
try: 
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell': args = parser.parse_args("") 
except:
    args = parser.parse_args()

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
z, _ = evaluate_sequential(model, data.float(), cond=cdata.float())
gaussian = pz.sample((100000, ))
density_2d(z.cpu().detach().numpy(), gaussian.cpu().detach().numpy(), 
            line_label=r'target gaussian $z$', hist_label=r'model(y, $\mathrm{cond}_\mathrm{Data}$)= $\hat{z}$', 
            xlim = [-3, 3], ylim = [-3, 3], save_as=args.output+'2d_gaussian_data.pdf')

z, _ = evaluate_sequential(model, mc.float(), cond=cmc.float())
gaussian = pz.sample((100000, ))
density_2d(z.cpu().detach().numpy(), gaussian.cpu().detach().numpy(), 
            line_label=r'target gaussian $z$', hist_label=r'model(y, $\mathrm{cond}_\mathrm{MC}$)= $\hat{z}$', 
            xlim = [-3, 3], ylim = [-3, 3], save_as=args.output+'2d_gaussian_mc.pdf')

# +

# Predict y

z = pz.sample((cmc.shape[0], ))
u, _ = evaluate_sequential(model, z, cmc.float(), rev=True)

u = u.cpu().detach().numpy()
u = input_scaler.inverse_transform(u)

data = input_scaler.inverse_transform(data)
mc = input_scaler.inverse_transform(mc)

# -

density_2d(u, data, 
            line_label=r'$u^\mathrm{MC}$', hist_label=r'model(z, $\mathrm{cond}_\mathrm{MC}$)=$\hat{u}$', 
            xlim = [-160, 100], ylim = [-80, 30], save_as=args.output+'2d_comp_umc_to_data.pdf')

density_2d(u, mc, 
            line_label=r'$u^\mathrm{MC}$', hist_label=r'model(z, $\mathrm{cond}_\mathrm{MC}$)=$\hat{u}$', 
            xlim = [-160, 100], ylim = [-80, 30], save_as=args.output+'2d_comp_umc_to_mc.pdf')





# # ### Compare MC -> Data

# u parallel
interval = [-170, 100]
_ = plt.hist(u[:,0], density=True, bins=100, range=interval, label=r'model(z,$c^\mathrm{MC}$)=$u_\parallel$')
_ = plt.hist(data[:,0], histtype=u'step', density=True, bins=100, 
                range=interval, linewidth=2, color='black', label=r'$u^\mathrm{Data}_\parallel$')
_ = plt.hist(dfmc['uP2_uncorrected'].values, histtype=u'step', density=True, bins=100, 
                range=interval, linewidth=2, color='red', label=r'$u^\mathrm{MC}_\perp $ uncorrected')
plt.xlabel(r'$u_\perp$')
plt.ylabel('a. u.')
plt.legend()
plt.savefig(args.output+'u_perp.pdf')
plt.clf()

# response
response(uperp=u[:,0], ptz=dfmc['pt_vis_c'], 
            save_path=args.output+'response.pdf', 
            cut_max=200, cut_min=25)


# n, bins, edges = plt.hist(up, histtype=r'step', bins=200, range=[-200, 200], 
#                             density=True, label=r'$\mathrm{u}_\parallel$')
# n1, bins1, edges1 = plt.hist(ptz, histtype=r'step', bins=200, range=[-200, 200], 
#                                 density=True, label=r'$p_\mathrm{T}^Z}\rangle$')
# plt.xlim([-200, 200])
# plt.legend()
# plt.xlabel('GeV')
# plt.ylabel('a.u.')
# plt.savefig(args.output+'uperp_ptw.pdf')
# plt.clf()


csingle_event = torch.concat([cmc[100,:].unsqueeze(0)]*10000)

# +
z = pz.sample((csingle_event.shape[0], ))
u, _ = evaluate_sequential(model, z, cond=csingle_event.float(), rev=True)

u = input_scaler.inverse_transform(u)
# -

density_2d(u, mc, 
            line_label=r'target $y_\mathrm{MC}$', hist_label=r'model(z, $\mathrm{cond}^\mathrm{fixed}_\mathrm{MC}$)=$\hat{y}$', 
            xlim = [-160, 100], ylim = [-80, 30], save_as=args.output+'fixedevent.pdf')

z = pz.sample((cdata[:100000,:].shape[0], ))
layerwise2d(model, z, cond=cdata[:100000,:], save_path=args.output+'layerwise.pdf', xlim=[-150, 100], ylim=[-80, 30], input_scaler=input_scaler)

for cond_no, cond_name in enumerate(cond):
    condition_correlation(model=model,
                            cond=cdata,
                            cond_no=cond_no,
                            deltas=np.linspace(-3, 3, 21),
                            input_scaler=input_scaler,
                            save_path=args.output+'cond_scan_' + cond_name + '.pdf',
                            cond_name= r'$\Delta$',
                            xlim=[-100, 80],
                            device=torch.device(args.output),
                            title = cond_name + r'+$\Delta\cdot \sigma_\mathrm(' + cond_name + ')$')
    print(cond_name, 'done')
