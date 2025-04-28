import torch
import torch.nn as nn

from JetTagger.utils.LorentzNet import LGEB_adapt as LGEB

class LorentzNet_Base(nn.Module):
    r''' Implimentation of LorentzNet.

    Args:
        - `n_scalar` (int): number of input scalars.
        - `n_hidden` (int): dimension of latent space.
        - `n_class`  (int): number of output classes.
        - `n_layers` (int): number of LGEB layers.
        - `c_weight` (float): weight c in the x_model.
        - `dropout`  (float): dropout rate.
    '''
    def __init__(self, in_channels, n_hidden, n_scalar, n_class = 2, n_layers = 6, c_weight = 1e-3, dropout = 0.):
        super(LorentzNet_Base, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_channels, n_hidden)
        self.LGEBs = nn.ModuleList([LGEB(
            self.n_hidden, 
            self.n_hidden, 
            self.n_hidden, 
            c_weight=c_weight, 
            last_layer=(i==n_layers-1)) for i in range(n_layers)])
        self.graph_dec = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.n_hidden, n_class)) # classification

    def forward(self, xp):
        s, p = xp
        h = self.embedding(s)

        for i in range(self.n_layers):
            h, p = self.LGEBs[i]([h, p, s])

        h = torch.mean(h, dim = 1)
        pred = self.graph_dec(h)
        return pred
    
class LorentzNet(LorentzNet_Base):
    def __init__(self, **kwargs) -> None:
        super(LorentzNet, self).__init__(
            **kwargs
        )
         


