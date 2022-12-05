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
#     display_name: cluster
#     language: python
#     name: python3
# ---

# +
import torch
import os
import numpy as np
from src.data import DataManager
import torch.distributions as dist

from src.helpers import calc_response, density_2d, ensure_dir, evaluate_sequential, plt_mlp_flow_with_errors
from src.models import get_flow_model, load_nn_ensemble
from argparse import ArgumentParser

# +
parser = ArgumentParser()
parser.add_argument("--test", dest="test", default=0, type=int, help='Test run {True, False}')
parser.add_argument("-c", "--cuda", dest="cuda", default=2, type=int, help='Cuda number')
parser.add_argument("--output", dest="output", default='flow_nnensemble_dummy/', type=str, help='Path for output files')
parser.add_argument("--model", dest="model", default='8flows_3layer_200nodes_50000batch/model.pt', type=str, help='Path to model')

parser.add_argument("-f", '--flows', dest="flows", default=8, type=int, help='Number of flows')
parser.add_argument("--nn-hidden", dest="nn_hidden", default=3, type=int, help='Number of hidden layers for the NN')
parser.add_argument("--nn-nodes", dest="nn_nodes", default=200, type=int, help='Number of hidden nodes for NN')

# +
# pass default arguments if executed as ipynb
try: 
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell': args = parser.parse_args("") 
except:
    args = parser.parse_args()
print(args)

ensure_dir(args.output)
device = torch.device(args.cuda)


#
# load data and models
#

dm = DataManager(args.test)

# mlps
paths = os.listdir('ensemble/')
mlps = load_nn_ensemble(paths, input_neurons=dm.cdata.shape[1], hidden_neurons=200, output_neurons=2, hiddenlayers=3)

# flows
model = get_flow_model(n_flows = args.flows, 
                       cond_shape = (dm.cmc.shape[1],),
                       nn_nodes = args.nn_nodes, 
                       nn_hidden = args.nn_hidden)
model.load_state_dict(torch.load(args.model, map_location="cpu"))


#
# one event multiple times
#

pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))

for event in range(100, 121):
    csingle_event = dm.cmc[event,:].unsqueeze(0)
    csingle_events = torch.concat([csingle_event]*10000)

    model.cpu()
    z = pz.sample((csingle_events.shape[0], ))
    u, jac = evaluate_sequential(model, z, cond=csingle_events.float(), rev=True)
    u = dm.input_scaler.inverse_transform(u)

    u_mlps = []
    for mlp in mlps:
        with torch.no_grad():
            out = mlp(csingle_event.float())
            u_mlps.append(out)
    u_mlps = torch.concat(u_mlps)
    u_mlps = dm.input_scaler.inverse_transform(u_mlps.numpy())

    density_2d(u, 
               u_mlps, 
               line_label=r'Ensemble NN($\mathrm{cond}^\mathrm{MC}$)=$\vec{u}$', 
               hist_label=r'model(z, $\mathrm{cond}^\mathrm{MC}$)=$\vec{u}$',
               crosses = [dm.dfmc["uP1_uncorrected"][event], 
                          dm.dfmc["uP2_uncorrected"][event]], 
               crosses_label=r'$\vec{u}_\mathrm{uncorrected}^\mathrm{MC}$',
               xlim = [-160, 100], 
               ylim = [-80, 30], 
               save_as=args.output+'fixedevent_flows_mlpensemble_eventid'+ str(event) +'.pdf', 
               gridsize=(50,50), 
               bins=100,
               hist_mean_label='model mean', 
               hist_mode_label='model mode', 
               grid=True)


#
# compare MLP ensemble with multiple evaluated NFlow
#

ptz = dm.dfmc['pt_vis_c']
dm.cmc = dm.cmc.to(device)

# Evaluate Mlp ensemble
rs = []
for mlp in mlps:
    with torch.no_grad():
        mlp.to(device)
        out = mlp(dm.cmc.float())
        out = dm.input_scaler.inverse_transform(out.cpu().numpy())
        out = torch.tensor(out[:,0]) # keep u_perp
        mlp.cpu()
    hist_raw, hist_weighted, bins, bin_mids = calc_response(uperp=out, ptz=ptz, cut_max=200, cut_min=25)
    r = hist_weighted/hist_raw
    rs.append(r)
rs = np.array(rs)

mlp_downs, mlp_ups = np.percentile(rs, q=[5, 95], axis=0)
mlp_means = np.mean(rs, axis=0)

# evaluate Flow multiple times
cmc = dm.cmc.to('cpu')
pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
rs_flow = []
flow_evaluations = 1

for i in range(len(mlps)):
    z = pz.sample((cmc.shape[0]*flow_evaluations, ))
    u_flow, _ = evaluate_sequential(model, z, cond=torch.vstack([cmc]*flow_evaluations).float(), 
                                    rev=True, device=device, batch=1000000)
    u_flow = u_flow.view(flow_evaluations, cmc.shape[0], u_flow.shape[1])
    u_flow = u_flow.mean(dim=0)
    
    u_flow = dm.input_scaler.inverse_transform(u_flow.cpu().numpy())
    hist_raw, hist_weighted, bins, bin_mids = calc_response(uperp=u_flow[:,0], ptz=ptz, cut_max=200, cut_min=25)
    r = hist_weighted/hist_raw
    rs_flow.append(r)

flow_downs, flow_ups = np.percentile(rs_flow, q=[5, 95], axis=0)
flow_means = np.mean(rs_flow, axis=0)

# plot comparison

plt_mlp_flow_with_errors(bin_mids, mlp_means, mlp_ups, mlp_downs, flow_means, 
                         flow_ups, flow_downs, bins, output_folder=args.output)

