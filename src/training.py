import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = [15, 10]
plt.rcParams.update({'axes.labelsize': 14})
matplotlib.use('AGG')


def empirical_risk(z, log_jac):
    # Empirical Risk Functional (KL Based)
    zz = torch.sum(z**2, dim=-1)
    neg_log_likeli = 0.5 * zz - log_jac
    loss = torch.mean(neg_log_likeli)
    return loss

class EarlyStopper(object):
    def __init__(self, stop, test=0):
        self.stop = stop
        self.steps_done = 0
        self.counter = 0
        self.best_loss = np.inf
        self.test = test
        
    def __call__(self):
        if self.stop < self.counter:
            print("Stop Training after {} steps".format(self.steps_done))
            print("Best loss: {:.4f}".format(self.best_loss))
            return False
        if self.steps_done >= self.test and self.test > 0:
            print("Stop test run after {} steps".format(self.steps_done))
            return False
        return True
        
    def step(self, loss, model):
        if loss >= self.best_loss: 
            self.counter+=1
        else:
            self.best_loss = loss
            self.counter = 0
            self.best_model_dict = model.state_dict()
        self.steps_done += 1

def plt_losses(losses, losses_val, folder, best_epoch):
    plt.plot(losses, label='training')
    plt.plot(losses_val, label='validation')
    plt.vlines(best_epoch, ymin=np.min(losses), ymax=np.max(losses), 
                label='best model', colors=['grey'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(folder+'loss.pdf')
