import uproot
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as dist

from unicodedata import decimal
from torch.utils.data import DataLoader, TensorDataset


def load_from_root(path, test=False):
    print('Reading files from: ', path)
    df = []
    for file in uproot.iterate(path+":ntuple", library="pd"):
        df.append(pd.DataFrame.from_dict(file))
        print('No. ', len(df))
        if test and len(df)>=1: break
    df = pd.concat(df)     
    return df

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def evaluate_sequential(model, data, cond, device=None, batch=5000, rev=False):
    with torch.no_grad():
        if device is None:
            device = torch.device('cpu')
        model = model.eval()
        dataset = TensorDataset(data, cond)
        loader = DataLoader(dataset, shuffle=False, batch_size=batch, num_workers=3, pin_memory=True)
        output = []
        jac = []
        
        model.to(device)
        for d, c in loader:
            c = c.to(device)
            d = d.to(device)
            out, j = model(d, c=[c], rev=rev)
            output.append(out)
            jac.append(j)
            
        jac = torch.concat(jac)
        output = torch.concat(output)
        return output, jac


#
#     Gaussian Density Plot
#

# +
def density_2d(hist, line, line_label='gaussian', hist_label='model(data)', xlim = [-3, 3], ylim = [-3, 3], save_as=None, xlabel=r'$u_\perp$', ylabel=r'$u_\parallel$'):
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'width_ratios': [2, 1],
                                                                'height_ratios': [1, 2]})
    plt.subplots_adjust(hspace=0, wspace=0)


    # heatmap
    ax[1,0].hexbin(x=hist[:,0], y=hist[:,1], extent= xlim + ylim,
                    label=hist_label, gridsize=(40,40), cmap='Blues', edgecolors=None)
    counts, ybins, xbins = np.histogram2d(line[:,0], line[:,1], 
                                            range=[xlim, ylim], bins=20)
    contours = ax[1,0].contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
                    colors='orange', levels=3)
    ax[1,0].plot([], [], '-', label=line_label, color='orange')
    ax[1,0].set_ylim(ylim)
    ax[1,0].set_xlim(xlim)
    ax[1,0].set_xlabel(xlabel)
    ax[1,0].set_ylabel(ylabel)
    ax[1,0].legend(loc='lower left', bbox_to_anchor=(1,1))

    # x axis
    _ = ax[0,0].hist(hist[:,0], bins=100, density=True, 
                        range=[xlim[0], xlim[1]], label=hist_label)
    _ = ax[0,0].hist(line[:,0], bins=100, range=[xlim[0], xlim[1]],
                        density=True, histtype=u'step', label=line_label)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_ylabel('a.u.')
    ax[0,0].set_xlim(xlim)

    # y axis
    _ = ax[1,1].hist(hist[:,1], bins=100, density=True, 
                        label=hist_label,range=[ylim[0], ylim[1]], 
                        orientation='horizontal')
    _ = ax[1,1].hist(line[:,1], bins=100, density=True, range=[ylim[0], ylim[1]],
                        histtype=u'step', label=line_label, orientation='horizontal')
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



def layerwise2d(model, z, cond, save_path=None, xlim=[None, None], ylim=[None, None], input_scaler=None, rev=True):

    length = len(model.module_list)
    columns = int(np.sqrt(length)) + 1
    
    fig, ax = plt.subplots(columns, columns)
    fig.set_figheight(3*columns)
    fig.set_figwidth(4*columns)
    
    z = z.detach().numpy()
    ax.flatten()[0].hexbin(z[:,0], z[:,1], cmap='rainbow')
    ax.flatten()[0].set_title('Input distribution', fontsize=10)
    
    cond = torch.tensor(cond)
    z = torch.tensor(z)
    
    iterator = range(length)
    if rev:
        iterator = reversed(iterator)

    for plot_no, nth_flow in enumerate(iterator):
        
        with torch.no_grad():
            z = model.module_list[nth_flow].forward(x=(z.float(), z.float()), c=[cond.float()], rev=rev)[0][0]
        
        z_ = z.detach().numpy()
        if input_scaler is not None:
            z_ = input_scaler.inverse_transform(z)

        ax.flatten()[plot_no+1].hexbin(z_[:,0], z_[:,1], cmap='rainbow', extent= xlim + ylim)
        ax.flatten()[plot_no+1].set_title('Output of layer #{}'.format(nth_flow+1), fontsize=10)
        
    for axis in ax.flatten()[length+1:]:
        axis.set_axis_off()
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()


def condition_correlation(model, 
                            cond, 
                            cond_no, 
                            deltas=[1, 0.4, 0, -0.4, -1], 
                            input_scaler=None,
                            save_path=None,
                            device=torch.device('cpu'),
                            xlim=[-250, 200],
                            xlabel=r'$u_\perp$ in GeV', 
                            cond_name=r'$x_i$',
                            title=None):

    pz = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
    z = pz.sample((cond.shape[0], ))

    out = []
    dd = np.abs(deltas[0] - deltas[1])/2.
    for delta in deltas:

        dummy = cond.clone()
        dummy[:,cond_no] += delta
        u, _ = evaluate_sequential(model, z, cond=dummy.float(), rev=True, device=device, batch=8000)
        u = u.to('cpu')
        if input_scaler is not None: 
            u = input_scaler.inverse_transform(u)
            u = torch.tensor(u)
        
        u[:,1] = torch.zeros_like(u[:,1]) + delta
        out.append(u)
        plt.hist2d(u[:,0].numpy(), u[:,1].numpy(), range=[xlim, [delta-dd, delta+dd]], bins=[100, 1], density=True)

    pltme = torch.concat(out)
    #ylim = [np.min(pltme[:,1].numpy())*1.05, np.max(pltme[:,1].numpy())*1.05]
    plt.ylim([deltas.min()-dd, deltas.max()+dd])
    
    #plt.hist2d(pltme[:,0].numpy(), pltme[:,1].numpy(), range=[xlim, ylim], bins=[100, len(deltas)])
    plt.plot([],[], label=r'model(z, cond$_\mathrm{Data}$)', color='None')
    plt.xlabel(xlabel)
    plt.ylabel(cond_name)
    plt.title(title)
    plt.legend()
    plt.xlim(xlim)
    if save_path is not None:
        plt.savefig(save_path)
        plt.clf()