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
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from src.helpers import ensure_dir, evaluate_sequential, density_2d, response, layerwise2d, condition_correlation
from src.models import get_flow_model
from src.data import DataManager, cond
from argparse import ArgumentParser

matplotlib.use('Agg')
plt.rcParams.update({'axes.labelsize': 14})

# +
parser = ArgumentParser()
parser.add_argument("--test", dest="test", default=0, type=int, help='Test run {True, False}')
parser.add_argument("-c", "--cuda", dest="cuda", default=0, type=int, help='Cuda number')
parser.add_argument("--output", dest="output", default='eval_dummy/', type=str, help='Path for output files')
parser.add_argument("--model", dest="model", default='8flows_3layer_200nodes_50000batch/model.pt', type=str, help='Path to model')

parser.add_argument("-f", '--flows', dest="flows", default=8, type=int, help='Number of flows')
parser.add_argument("--nn-hidden", dest="nn_hidden", default=3, type=int, help='Number of hidden layers for the NN')
parser.add_argument("--nn-nodes", dest="nn_nodes", default=200, type=int, help='Number of hidden nodes for NN')

# pass default arguments if executed as ipynb
try: 
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell': args = parser.parse_args("") 
except:
    args = parser.parse_args()
print(args)

ensure_dir(args.output)

# 
# load data and model
#

dm = DataManager(args.test)

model = get_flow_model(n_flows = args.flows, 
                       cond_shape = (dm.cmc.shape[1],),
                       nn_nodes = args.nn_nodes, 
                       nn_hidden = args.nn_hidden)

model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
model.cpu()


#
# model -> gaussian
#
'''
z, _ = evaluate_sequential(model, dm.data.float(), cond=dm.cdata.float())

pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
gaussian = pz.sample((100000, ))

density_2d(hist=z.cpu().detach().numpy(), 
           line=gaussian.cpu().detach().numpy(), 
           line_label=r'target Gaussian $z$', 
           hist_label=r'model($\vec{u}^\mathrm{Data}$, $\mathrm{cond}^\mathrm{Data}$)= $\hat{z}$', 
           xlim = [-3, 3], 
           ylim = [-3, 3],
           xlabel =r'x',
           ylabel = r'y',
           save_as=args.output+'2d_gaussian_data.pdf')


z, _ = evaluate_sequential(model, dm.mc.float(), cond=dm.cmc.float())
gaussian = pz.sample((100000, ))
density_2d(hist=z.cpu().detach().numpy(), 
           line=gaussian.cpu().detach().numpy(), 
           line_label=r'target Gaussian $z$', 
           hist_label=r'model($\vec{u}^\mathrm{MC}$, $\mathrm{cond}^\mathrm{MC}$)= $\hat{z}$', 
           xlim = [-3, 3],
           ylim = [-3, 3],
           xlabel =r'x',
           ylabel = r'y',
           save_as=args.output+'2d_gaussian_mc.pdf')
'''

#
# model -> u
#

pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
z = pz.sample((dm.cmc.shape[0], ))
#u, _ = evaluate_sequential(model, z, dm.cmc.float(), rev=True)
with torch.no_grad():
    u, j = model(z, c=[dm.cmc.float()], rev=True)
print("pred. done")
u = u.cpu().detach().numpy()
u = dm.input_scaler.inverse_transform(u)

data = dm.input_scaler.inverse_transform(dm.data)
mc = dm.input_scaler.inverse_transform(dm.mc)
print('rescaling done')


density_2d(hist=u, 
           line=data, 
           line_label=r'$\vec{u}^\mathrm{Data}_\mathrm{uncorrected}$', 
           hist_label=r'model(z, $\mathrm{cond}^\mathrm{MC}$)=$\hat{\vec{u}}$', 
           xlim = [-160, 100], 
           ylim = [-80, 30],
           gridsize=(20,20),
           save_as=args.output+'2d_comp_umc_to_data.pdf')

density_2d(hist=u,
           line=mc, 
           line_label=r'$\vec{u}^\mathrm{MC}_\mathrm{uncorrected}$', 
           hist_label=r'model(z, $\mathrm{cond}^\mathrm{MC}$)=$\hat{\vec{u}}$', 
           xlim = [-160, 100], 
           ylim = [-80, 30],
           gridsize=(20,20),
           save_as=args.output+'2d_comp_umc_to_mc.pdf')
print("done")

#
#  Compare MC -> Data
#

# u parallel
interval = [-170, 100]
_ = plt.hist(u[:,1], 
             density=True, 
             bins=100, 
             range=interval, 
             label=r'model(z,$c^\mathrm{MC}$)=$u_\perp$')
_ = plt.hist(data[:,1], 
             histtype=u'step', 
             density=True, 
             bins=100, 
             range=interval, 
             linewidth=2, 
             color='black', 
             label=r'$u^\mathrm{Data}_\perp$')
_ = plt.hist(dm.dfmc['uP2_uncorrected'].values, 
             histtype=u'step', 
             density=True, 
             bins=100, 
             range=interval, 
             linewidth=2, 
             color='red',
             label=r'$u^\mathrm{MC}_\perp $ uncorrected')

plt.xlabel(r'$u_\perp$')
plt.ylabel('a. u.')
plt.legend()
plt.savefig(args.output+'u_perp.pdf')
plt.clf()

#
# others
#

#response
response(uperp=u[:,0], 
        ptz=dm.dfmc['pt_vis_c'], 
        save_path=args.output+'response.pdf', 
        cut_max=200, cut_min=25)

response(uperp=dm.dfmc['uP1_uncorrected'].values, 
        ptz=dm.dfmc['pt_vis_c'], 
        save_path=args.output+'response_test.pdf', 
        cut_max=200, cut_min=25)

'''
# look at single event
csingle_event = torch.concat([dm.cmc[100,:].unsqueeze(0)]*10000)
z = pz.sample((csingle_event.shape[0], ))
u, _ = evaluate_sequential(model, z, cond=csingle_event.float(), rev=True)
u = dm.input_scaler.inverse_transform(u)

density_2d(u, 
           mc, 
           line_label=r'target $y_\mathrm{MC}$', 
           hist_label=r'model(z, $\mathrm{cond}_\mathrm{fixed}^\mathrm{MC}$)=$\hat{y}$', 
           xlim = [-160, 100], 
           ylim = [-80, 30], 
           save_as=args.output+'fixedevent.pdf')

# block wise introspection
z = pz.sample((dm.cdata[:100000,:].shape[0], ))
layerwise2d(model, 
            z, 
            cond=dm.cdata[:100000,:], 
            save_path=args.output+'layerwise.pdf', 
            xlim=[-150, 100], 
            ylim=[-80, 30], 
            input_scaler=dm.input_scaler)

# correlation on conditions
for cond_no, cond_name in enumerate(cond):
    condition_correlation(model=model,
                            cond=dm.cdata,
                            cond_no=cond_no,
                            deltas=np.linspace(-3, 3, 21),
                            input_scaler=dm.input_scaler,
                            save_path=args.output+'cond_scan_' + cond_name + '.pdf',
                            cond_name= r'$\Delta$',
                            xlim=[-100, 80],
                            device=torch.device(args.cuda),
                            title = cond_name + r'+$\Delta\cdot \sigma_{}$'.format("{"+cond_name+"}"))
    print(cond_name, 'done')
'''