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
import torch.distributions as dist
import torch.optim as optim
import os
import wandb

from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser

from src.models import get_flow_model
from src.helpers import ensure_dir
from src.data import DataManager
from src.training import EarlyStopper, empirical_risk, plt_losses

# +
parser = ArgumentParser()
parser.add_argument("-c", "--cuda", dest="cuda", default=0, type=int, help='Cuda to use. In case of Multiprocessing: main cuda number')
parser.add_argument("--ndevices", dest="ndevices", default=1, type=int, help='Number of devices for Multiprocessing') # count_devices() and os.environ['CUDA_VISIBLE_DEVICES'] are not reliable on the batch system
parser.add_argument("--test", dest="test", default=1, type=int, help='Test run {True, False}')

parser.add_argument("--batch", dest="batch", default=50000, type=int, help='Batch size')
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help='Learning rate')
parser.add_argument("-f", '--flows', dest="flows", default=2, type=int, help='Number of flows')
parser.add_argument("--nn-hidden", dest="nn_hidden", default=2, type=int, help='Number of hidden layers for the NN')
parser.add_argument("--nn-nodes", dest="nn_nodes", default=150, type=int, help='Number of hidden nodes for NN')

# pass default arguments if executed as ipynb
try: 
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell': args = parser.parse_args("") 
except:
    args = parser.parse_args()
print(args)

folder = 'output/'
ensure_dir(folder)

# initialize wandb
wandb.config = {"batch_size": args.batch}
wandb.init(project="my-test-project")

#
# load ddata & model
#

dm = DataManager(args.test)

device = torch.device(args.cuda)
dm.data_val = dm.data_val.to(device)
dm.cdata_val = dm.cdata_val.to(device)

# get Model
model = get_flow_model(n_flows = args.flows, 
                       cond_shape = (dm.cmc.shape[1],),
                       nn_nodes = args.nn_nodes, 
                       nn_hidden = args.nn_hidden)

# +
use_dataparallel = os.uname()[1] != 'deepthought' and args.ndevices > 1
if use_dataparallel:
    print("Use DataParallel on {} GPUs".format(args.ndevices))
    model = nn.DataParallel(model, device_ids=list(range(0,args.ndevices)))
# -

#
# Training
#

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, 
                                                    patience=3, verbose=True)
dataset = TensorDataset(dm.data, dm.cdata)
loader = DataLoader(dataset, shuffle=True, batch_size=args.batch, num_workers=4)

# +
pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
model.to(device)
stopper = EarlyStopper(stop=15, test=args.test)
nbatches = len(loader)

losses = []
losses_val = []

while stopper():
    epo_loss = 0
    for d, c in loader:
        d = d.to(device)
        c = c.to(device)

        z, log_jac = model(d.float(), c=[c.float()])
        loss = empirical_risk(z, log_jac)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epo_loss += loss.cpu().detach().numpy()
        
    epo_loss = round(epo_loss/nbatches,5)
    scheduler.step(epo_loss)
    losses.append(epo_loss)
    stopper.step(epo_loss, model)
    
    # validation
    z, log_jac = model(dm.data_val.float(), c=[dm.cdata_val.float()])
    loss_val = empirical_risk(z, log_jac)
    loss_val = loss_val.cpu().detach().numpy()
    losses_val.append(loss_val)
    
    print('Epoch: {:.0f}; train Loss: {:.3f}; val Loss: {:.3f}; stopper: {:.0f}; best val loss: {:.3f}\n'.format(stopper.steps_done, epo_loss, loss_val, stopper.counter, stopper.best_loss))

    # logging
    wandb.log({"loss_train": epo_loss, 
               "loss_val": loss_val, 
               "batch": args.batch, 
               'lr': optimizer.param_groups[0]['lr']})
    #wandb.watch(model)

# -
plt_losses(losses, losses_val, folder, stopper.counter - stopper.stop)
model.load_state_dict(stopper.best_model_dict)

if use_dataparallel:
    model = model.module # unwrap from DataParallel

torch.save(model.state_dict(), folder+'model.pt')
os.system('cp ' + os.path.basename(__file__) + ' ' + folder + 'code.py') 
print('Training done')