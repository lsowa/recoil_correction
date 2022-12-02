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
import torch.optim as optim

from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.helpers import ensure_dir
from src.data import DataManager
from src.models import Mlp
from src.training import EarlyStopper, plt_losses

# +

parser = ArgumentParser()
parser.add_argument("--test", dest="test", default=0, type=int, help='Test run {True, False}')
parser.add_argument("-c", "--cuda", dest="cuda", default=0, type=int, help='Cuda number')
parser.add_argument("--output", dest="output", default='nn_dummy/', type=str, help='Path for output files')

parser.add_argument("--nn-hidden", dest="nn_hidden", default=3, type=int, help='Number of hidden layers for the NN')
parser.add_argument("--nn-nodes", dest="nn_nodes", default=200, type=int, help='Number of hidden nodes for NN')
parser.add_argument("--lr", dest="lr", default=0.01, type=float, help='Learning rate (to start with)')
parser.add_argument("--batch", dest="batch", default=500, type=int, help='Batch Size')
args = parser.parse_args()

# pass default arguments if executed as ipynb
try: 
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell': args = parser.parse_args("") 
except:
    args = parser.parse_args()

ensure_dir(args.output)
device = args.cuda


#
# Load data and model
#

dm = DataManager(args.test)

model = Mlp(input_neurons=dm.cdata.shape[1], 
            hidden_neurons=args.nn_nodes, 
            output_neurons=2, 
            hiddenlayers=args.nn_hidden)

#
# model training
#

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, 
                                                 patience=3, verbose=True)
mse = nn.MSELoss()

dataset = TensorDataset(dm.data, dm.cdata)
loader = DataLoader(dataset, shuffle=True, 
                    batch_size=args.batch, 
                    num_workers=4, 
                    pin_memory=True)

model.to(device)
dm.cdata_val = dm.cdata_val.to(device)
dm.data_val = dm.data_val.to(device)

losses = []
losses_val = []
nbatches = len(loader)

stopper = EarlyStopper(stop=15, test=args.test)
while stopper():
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
        
    epo_loss = round(epo_loss/nbatches,5)
    scheduler.step(epo_loss)
    stopper.step(epo_loss, model)
    losses.append(epo_loss)
    
    # validation
    u = model(dm.cdata_val.float())
    loss_val = mse(u.float(), dm.data_val.float())
    loss_val = loss_val.cpu().detach().numpy()
    losses_val.append(loss_val)
    
    print('Epoch: {:.0f}; train Loss: {:.3f}; val Loss: {:.3f}; stopper: {:.0f}; best val loss: {:.3f}\n'.format(stopper.steps_done, 
                                                                                                                 epo_loss, loss_val, 
                                                                                                                 stopper.counter, 
                                                                                                                 stopper.best_loss))

plt_losses(losses, losses_val, args.output, stopper.counter - stopper.stop)
model.load_state_dict(stopper.best_model_dict)

torch.save(model.state_dict(), args.output+'model.pt')


