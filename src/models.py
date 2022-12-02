import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch

def get_flow_model(n_flows, cond_shape, nn_nodes, nn_hidden):

    def mlp_constructor(input_dim=2, out_dim=2, hidden_nodes=nn_nodes, nn_hidden=nn_hidden):
        layers = [nn.Linear(input_dim, hidden_nodes), nn.ReLU()]
        for n in range(nn_hidden-1):
            layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_nodes, out_dim))
        model = nn.Sequential(*layers)
        return model

    model = Ff.SequenceINN(2)
    for k in range(n_flows):
        model.append(Fm.RNVPCouplingBlock, subnet_constructor=mlp_constructor, 
                        clamp=2, cond=0, cond_shape=cond_shape)
    return model


class Mlp(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, hiddenlayers):

        nn.Module.__init__(self)

        # mlp layers
        self.mlplayers = nn.ModuleList([nn.Linear(input_neurons, hidden_neurons)])
        self.mlplayers.extend([nn.Linear(hidden_neurons, hidden_neurons) for i in range(hiddenlayers + 1)])
        self.mlplayers.append(nn.Linear(hidden_neurons, output_neurons))

    def forward(self, x):
        # input shape: (batch, features)
        for mlplayer in self.mlplayers[:-1]:
            x = mlplayer(x)
            x = torch.tanh(x)

        # new x: (batch, 1)
        x = self.mlplayers[-1](x)
        x = x.squeeze(-1)  # new x: (batch)
        return x

def load_nn_ensemble(paths, input_neurons, hidden_neurons, output_neurons, hiddenlayers):
    mlps = []
    for path in paths:
        with torch.no_grad():
            mlp = Mlp(input_neurons=input_neurons, 
                      hidden_neurons=hidden_neurons, 
                      output_neurons=output_neurons, 
                      hiddenlayers=hiddenlayers)
            path = 'ensemble/' + path + '/model.pt'
            mlp.load_state_dict(torch.load(path, map_location="cpu"))
            mlps.append(mlp)
    return mlps