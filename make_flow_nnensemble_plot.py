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
import matplotlib.pyplot as plt
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import matplotlib

from helpers import *
from os.path import exists
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')

# +
parser = ArgumentParser()
parser.add_argument("-c", "--cuda", dest="cuda", default=1, type=int, help='Cuda number')

parser.add_argument("-f", '--flows', dest="flows", default=8, type=int, help='Number of flows')
parser.add_argument("--nn-hidden", dest="nn_hidden", default=3, type=int, help='Number of hidden layers for the NN')
parser.add_argument("--nn-nodes", dest="nn_nodes", default=200, type=int, help='Number of hidden nodes for NN')
parser.add_argument("--model", dest="model", default='8flows_3layer_200nodes_50000batch/model.pt', type=str, help='Path to model')
parser.add_argument("--output", dest="output", default='flow_nnensemble_dummy/', type=str, help='Path for output files')

parser.add_argument("--test", dest="test", default=0, type=int, help='Test run {True, False}')

# +
# pass default arguments if executed as ipynb
try: 
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell': args = parser.parse_args("") 
except:
    args = parser.parse_args()

device = torch.device(args.cuda)
print(args)
ensure_dir(args.output)
# -

cond = ['metphi','pt_vis_c', 'phi_vis_c','pt_1', 'pt_2','dxy_1', 'dxy_2','dz_1',
        'dz_2','eta_1', 'eta_2','mass_1', 'mass_2','metSumEt']
names = ['uP1_uncorrected', 'uP2_uncorrected']
paths = os.listdir('ensemble/')

if exists('/ceph/lsowa/recoil/dt.root'):
    dfdata = load_from_root('/ceph/lsowa/recoil/dt.root', test=args.test)
    dfmc = load_from_root('/ceph/lsowa/recoil/mc.root', test=args.test)
else:
    # when running on cluster
    dfdata = load_from_root('recoil/dt.root', test=args.test)
    dfmc = load_from_root('recoil/mc.root', test=args.test)

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

pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))

# +
#
# Load models
#
# -

# mlps
mlps = []
for path in paths:
    with torch.no_grad():
        mlp = Mlp(input_neurons=cdata.shape[1], hidden_neurons=200, output_neurons=2, hiddenlayers=3)
        path = 'ensemble/' + path + '/model.pt'
        mlp.load_state_dict(torch.load(path, map_location="cpu"))
        mlps.append(mlp)


# +
# flows
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

model.load_state_dict(torch.load(args.model, map_location="cpu"))

# +
#
# one event multiple times
#
# -

csingle_event = cmc[100,:].unsqueeze(0)
csingle_events = torch.concat([csingle_event]*10000)

model.cpu()

# +
z = pz.sample((csingle_events.shape[0], ))
u, _ = evaluate_sequential(model, z, cond=csingle_events.float(), rev=True)

u = input_scaler.inverse_transform(u)
# -

u_mlps = []
for mlp in mlps:
    with torch.no_grad():
        out = mlp(csingle_event.float())
        u_mlps.append(out)
u_mlps = torch.concat(u_mlps)
u_mlps = input_scaler.inverse_transform(u_mlps.numpy())

density_2d(u, u_mlps, 
            line_label=r'NN Ensamble NN(cond)$', hist_label=r'model(z, $\mathrm{cond}^\mathrm{fixed}_\mathrm{MC}$)=$\hat{y}$', 
            xlim = [-160, 100], ylim = [-80, 30], save_as=args.output+'fixedevent_flows_mlpensemble.pdf', gridsize=(50,50), bins=100)

# +
#
# compare MLP ensemble with multiple evaluated NFlow
#
# -

ptz = dfmc['pt_vis_c']
cmc = cmc.to(device)

# Evaluate Mlp ensemble
rs = []
for mlp in mlps:
    with torch.no_grad():
        mlp.to(device)
        out = mlp(cmc.float())
        out = input_scaler.inverse_transform(out.cpu().numpy())
        out = torch.tensor(out[:,0]) # keep u_perp
    hist_raw, hist_weighted, bins, bin_mids = calc_response(uperp=out, ptz=ptz, cut_max=200, cut_min=25)
    r = hist_weighted/hist_raw
    rs.append(r)
rs = np.array(rs)

mlp_downs, mlp_ups = np.percentile(rs, q=[2.5, 97.5], axis=0)
mlp_means = np.mean(rs, axis=0)

# evaluate NFlow multiple times
cmc = cmc.to('cpu')

pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
rs_flow = []
for i in range(len(mlps)):
    z = pz.sample((cmc.shape[0], ))
    u_flow, _ = evaluate_sequential(model, z, cond=cmc.float(), rev=True, device=device)
    u_flow = input_scaler.inverse_transform(u_flow.cpu().numpy())
    hist_raw, hist_weighted, bins, bin_mids = calc_response(uperp=u_flow[:,0], ptz=ptz, cut_max=200, cut_min=25)
    r = hist_weighted/hist_raw
    rs_flow.append(r)

flow_downs, flow_ups = np.percentile(rs_flow, q=[2.5, 97.5], axis=0)
flow_means = np.mean(rs_flow, axis=0)

# +
plt.errorbar(x=bin_mids, y=mlp_means,
            xerr=(bins[1:]-bins[:-1])/2, yerr=[mlp_means-mlp_downs, mlp_ups-mlp_means], fmt='o', capsize=2, label='MLP Ensemble')
plt.errorbar(x=bin_mids, y=flow_means,
            xerr=(bins[1:]-bins[:-1])/2, yerr=[flow_means-flow_downs, flow_ups-flow_means], fmt='o', capsize=2, label='Flow')

plt.hlines([1], bins.min(), bins.max(), color='black')
plt.xlabel(r'$p_\mathrm{T}^Z$ in GeV')
plt.ylabel(r'$\langle \frac{\mathrm{u}_\parallel}{p_\mathrm{T}^Z}\rangle$')
plt.xlim([bins.min(), bins.max()])
plt.legend()
plt.savefig(args.output + 'response_compare.pdf')
plt.clf()

