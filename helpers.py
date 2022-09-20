import uproot
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from unicodedata import decimal
from torch.utils.data import DataLoader, TensorDataset

def load_from_root(path, test=False):
    print('Reading: ', path)
    df = []
    for file in uproot.iterate(path+":ntuple", library="pd"):
        df.append(pd.DataFrame.from_dict(file))
        print(len(df))
        if test and len(df)>=1: break
    df = pd.concat(df)     
    return df

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate_sequential(model, data, cond, device=None, batch=5000, rev=False):
    
    if device is None:
        device = torch.device('cpu')
    
    dataset = TensorDataset(data, cond)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch)
    output = []
    
    model.to(device)
    for d, c in loader:
        c = c.to(device)
        d = d.to(device)
        out, _ = model(d, c=[c], rev=rev)
        output.append(out)
    
    output = torch.concat(output)
    return output



#
#     Gaussian Density Plot
#

# +
def density_2d(data, dist, dist_label='gaussian', data_label='model(data)', xlim = [-3, 3], ylim = [-3, 3], save_as=None):
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'width_ratios': [2, 1],
                                                                'height_ratios': [1, 2]})
    plt.subplots_adjust(hspace=0, wspace=0)


    # heatmap
    ax[1,0].hexbin(x=data[:,0], y=data[:,1], extent= xlim + ylim,
                    label=data_label, gridsize=(40,40), cmap='Blues', edgecolors=None)
    counts, ybins, xbins = np.histogram2d(dist[:,0], dist[:,1], 
                                            range=[xlim, ylim], bins=20)
    contours = ax[1,0].contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
                    colors='orange', levels=3)
    ax[1,0].plot([], [], '-', label=dist_label, color='orange')
    ax[1,0].set_ylim(ylim)
    ax[1,0].set_xlim(xlim)
    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    ax[1,0].legend(loc='lower left', bbox_to_anchor=(1,1))

    # x axis
    _ = ax[0,0].hist(data[:,0], bins=100, density=True, 
                        range=[xlim[0], xlim[1]], label=data_label)
    _ = ax[0,0].hist(dist[:,0], bins=100, range=[xlim[0], xlim[1]],
                        density=True, histtype=u'step', label=dist_label)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_ylabel('a.u.')
    ax[0,0].set_xlim(xlim)

    # y axis
    _ = ax[1,1].hist(data[:,1], bins=100, density=True, 
                        label=data_label,range=[ylim[0], ylim[1]], 
                        orientation='horizontal')
    _ = ax[1,1].hist(dist[:,1], bins=100, density=True, range=[ylim[0], ylim[1]],
                        histtype=u'step', label=dist_label, orientation='horizontal')
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[1,1].set_xlabel('a.u.')
    ax[1,1].set_ylim(ylim)

    # third wheel
    _ = ax[0,1].axis('off')
    
    if save_as is not None:
        plt.savefig(save_as)
        plt.clf()
    # -

