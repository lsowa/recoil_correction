import os
import torch

import numpy as np
import torch.distributions as dist
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

matplotlib.rcParams['figure.figsize'] = [15, 10]
plt.rcParams.update({'axes.labelsize': 14})
matplotlib.use('AGG')

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

def get_max_from_hist(n, bins):
    mids = bins[:-1] + (bins[2] - bins[1])/2
    n_max = n.argmax()
    return mids[n_max]


def density_2d(hist, 
               line, 
               line_label='gaussian', 
               hist_label='model(data)', 
               xlim = [-3, 3], 
               ylim = [-3, 3], 
               xlabel=r'$u_\parallel$',
               ylabel=r'$u_\perp$',
               crosses = None, 
               crosses_label = None, 
               crosses_color='red',
               gridsize=(40,40), 
               bins=100, 
               levels=4, 
               alpha=0.7,
               hist_mean_label=None, 
               hist_mode_label=None, 
               grid=False,
               save_as=None):
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'width_ratios': [2, 1],
                                                                'height_ratios': [1, 2]})
    plt.subplots_adjust(hspace=0, wspace=0)

    # x axis
    nx, binsx, _ = ax[0,0].hist(hist[:,0], bins=bins, density=True, 
                        range=[xlim[0], xlim[1]], label=hist_label)
    _ = ax[0,0].hist(line[:,0], bins=bins, range=[xlim[0], xlim[1]],
                        density=True, histtype=u'step', label=line_label)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_ylabel('a.u.')
    ax[0,0].set_xlim(xlim)

    # y axis
    ny, binsy, _ = ax[1,1].hist(hist[:,1], bins=bins, density=True, 
                        label=hist_label,range=[ylim[0], ylim[1]], 
                        orientation='horizontal')
    _ = ax[1,1].hist(line[:,1], bins=bins, density=True, range=[ylim[0], ylim[1]],
                        histtype=u'step', label=line_label, orientation='horizontal')
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[1,1].set_xlabel('a.u.')
    ax[1,1].set_ylim(ylim)


    # heatmap
    ax[1,0].hexbin(x=hist[:,0], y=hist[:,1], extent= [-50, 50] + [-25, 25],#xlim + ylim,
                    label=hist_label, gridsize=gridsize, cmap='Blues', edgecolors=None)

    sns.kdeplot(x=line[:,0], y=line[:,1], shade=False, 
                ax=ax[1,0], color='orange', levels=levels, alpha=alpha)
    ax[1,0].plot([], [], '-', label=line_label, color='orange')
    
    if crosses is not None:
        ax[1,0].plot(crosses[0], crosses[1], '*', label=crosses_label, color=crosses_color)
    if hist_mean_label is not None:
        ax[1,0].plot(hist[:,0].mean(), hist[:,1].mean(), '+', label=hist_mean_label, color="green")
    if hist_mode_label is not None:
        ax[1,0].plot(get_max_from_hist(nx, binsx), get_max_from_hist(ny, binsy), 
                        '+', label=hist_mode_label, color="magenta")

    if grid:
        ax[1,0].grid()
        
    ax[1,0].set_ylim(ylim)
    ax[1,0].set_xlim(xlim)
    ax[1,0].set_xlabel(xlabel)
    ax[1,0].set_ylabel(ylabel)
    ax[1,0].legend(loc='lower left')#, bbox_to_anchor=(1,1))
    
    # third wheels
    _ = ax[0,1].axis('off')

    if save_as is not None:
        plt.tight_layout()
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
        plt.tight_layout()
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
        u, _ = evaluate_sequential(model, z, cond=dummy.float(), rev=True, device=device, batch=4000)
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
        plt.tight_layout()
        plt.savefig(save_path)
        plt.clf()
        

def calc_response(uperp, ptz, cut_max=200, cut_min=25):
    keep = np.logical_and(cut_max > ptz, ptz > cut_min)
    ptz = ptz[keep]
    uperp = uperp[keep]
    r = -uperp/ptz

    bins = np.linspace(cut_min, cut_max, 10) # edges
    bin_mids = (bins[1:]-bins[:-1])/2 + bins[:-1] # mids
    
    hist_raw, edges = np.histogram(ptz, bins = bins, range=[cut_min, cut_max])
    hist_weighted, edges = np.histogram(ptz, bins = bins, weights=r, range=[cut_min, cut_max])
    return hist_raw, hist_weighted, bins, bin_mids



def response(uperp, ptz, save_path=None, cut_max=200, cut_min=25):
    
    hist_raw, hist_weighted, bins, bin_mids = calc_response(uperp, ptz, cut_max=cut_max, cut_min=cut_min)
    plt.errorbar(x=bin_mids, y=hist_weighted/hist_raw,
                xerr=(bins[1:]-bins[:-1])/2, fmt='o', capsize=2)

    plt.hlines([1], bins.min(), bins.max(), color='black')
    plt.xlabel(r'$p_\mathrm{T}^Z$ in GeV')
    plt.ylabel(r'$\langle \frac{\mathrm{u}_\parallel}{p_\mathrm{T}^Z}\rangle$')
    plt.xlim([bins.min(), bins.max()])
    plt.savefig(save_path)
    plt.clf()

def plt_mlp_flow_with_errors(bin_mids, mlp_means, mlp_ups, mlp_downs, flow_means, flow_ups, flow_downs, bins, output_folder,
                             mlp_label='MLP Ensemble 90% Conf. Int.', 
                             flow_label='Mode of Flow 90% Conf. Int.'):
    
    plt.errorbar(x=bin_mids, y=mlp_means,
                xerr=(bins[1:]-bins[:-1])/2, yerr=[mlp_means-mlp_downs, mlp_ups-mlp_means], fmt='o', capsize=2, label=mlp_label)
    plt.errorbar(x=bin_mids, y=flow_means,
                xerr=(bins[1:]-bins[:-1])/2, yerr=[flow_means-flow_downs, flow_ups-flow_means], fmt='o', capsize=2, label=flow_label)

    plt.hlines([1], bins.min(), bins.max(), color='black')
    plt.xlabel(r'$p_\mathrm{T}^Z$ in GeV')
    plt.ylabel(r'$\langle \frac{\mathrm{u}_\parallel}{p_\mathrm{T}^Z}\rangle$')
    plt.xlim([bins.min(), bins.max()])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder + 'response_compare_new.pdf')