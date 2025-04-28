import torch
import torch.nn as nn
from typing import Union, Tuple

class MomentumFormerLayer(nn.Module):
    """
    MomentumFormerLayer implements a vector-based attention mechanism for particle cloud data, integrating momentum-involved interactions, equivariant positional encoding, and momentum propagation updating.
    
    Args:
    
        in_planes (int): Dimension of input features.
        out_planes (int): Dimension of output features after attention.
        share_planes (int, optional): Number of shared channels during interaction (unused in this layer). Default is 8. (deprecated)
        nsample (int, optional): Number of local neighbors considered (used in context, not directly here). Default is 16. (deprecated)
        mode (str, optional): Interaction mode (reserved for variations). Default is 'Int'. (deprecated)
        return_attn (bool, optional): Whether to return the raw attention map along with outputs. Default is False.
        
    """
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16, mode = 'Int', return_attn: bool = False):
        super().__init__()
        self.out_planes = out_planes
        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_w = nn.Sequential(
            nn.LayerNorm(out_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes, out_planes),
            nn.LayerNorm(out_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes, out_planes),
        )
        self.softmax = nn.Softmax(dim=1)
        #self.mode = mode 
        
        self.equivariant_pe = nn.Sequential(
            nn.Linear(4, self.out_planes//4, False),
            nn.LayerNorm(self.out_planes//4, elementwise_affine = False, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(self.out_planes//4, self.out_planes, False),
            nn.LayerNorm(self.out_planes, elementwise_affine = False, bias = False),
            nn.ReLU(inplace = True),
        )
        
        self.phi_e = self.prepare_phi_e(in_channels = self.out_planes * 2 + 1, out_channels = self.out_planes)
        self.phi_x = self.prepare_phi_x(in_channels = self.out_planes, hidden = 32) 
        self.phi_h = self.prepare_phi_h(out_channels = self.out_planes)
        self.return_attn = return_attn 
    
    def prepare_phi_e(self, in_channels: int, out_channels: int):
        phi_e = nn.Sequential(
            nn.Linear(in_channels, out_channels//4, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(out_channels//4, out_channels, bias = False),
            nn.ReLU(inplace = True)
        )
        
        
        return phi_e
    
    def prepare_phi_h(self, out_channels: int):
        phi_h = nn.Sequential(
            nn.Linear(out_channels, out_channels, False),
            nn.ReLU(inplace = True)
        )
        return phi_h
    
    def prepare_phi_x(self, in_channels: int, hidden: int):
        
        phi_x = nn.Sequential(
            nn.LayerNorm(in_channels, elementwise_affine = False, bias = False),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace = True),
            nn.Linear(hidden, 1),
            nn.ReLU(inplace = True)
        ) 
        
        return phi_x 
     
    def forward(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for MomentumFormerLayer.
        
        This layer processes particle features and momentum by:
        - computing vector-based attention accross particle pairs using momentum-involved interaction.
        - Updating particle features through weighted aggregation.
        - Updating particle momenta based on learned inter-particle momentum propogation weights
        
        Args:
            input (dict):
                A dictionary containing:
                    - x (torch.Tensor): Particle Features of shape (N, P, C_in)
                    - p (torch.Tensor): Particle four momentum of shape (N, P, 4)
                    - U (torch.Tensor): Particle pairwise interaction embeddings of shape (N, P, P, 4)
        
        """
        
        
        x, p, U = input 
        x_k, x_v, x_q =  self.linear_k(x), self.linear_v(x), self.linear_q(x)
        
        Ck = x_k.shape[-1]
        Cq = x_q.shape[-1]
        n_pts = x_k.shape[1]
        x_k = x_k.unsqueeze(2).expand(-1, -1, n_pts, Ck)
        x_q = x_q.unsqueeze(1).expand(-1, n_pts, -1, Cq)
        pi = p.unsqueeze(2).expand(-1, -1, n_pts, -1)
        pj = p.unsqueeze(1).expand(-1, n_pts, -1, -1)
        
        key_padding_mask = (p[:, :, 0] == 0.)
        key_padding_mask = key_padding_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_pts, self.out_planes) 
        registered_mask = torch.sum(p[:, :, 0] != 0., dim = -1).unsqueeze(-1)
        
        EQPE = self.equivariant_pe(pi-pj)
         
        x_v = x_v.unsqueeze(1).expand(-1, n_pts, -1, -1)
        r_qk = x_k - x_q + U + EQPE
        x_v = x_v + U + EQPE
        w = self.linear_w(r_qk) # each element in weight matrix corresponds to each element in r_qk  -> element: vector weight: w_ij * v_jc, for i,j the index of point, c the index of vector element
        w = w.masked_fill(key_padding_mask, -1e4)
        Ck = w.shape[-1]
        N, pt1, pt2 = w.shape[0], w.shape[1], w.shape[2]
        w = w.view(-1, pt2, Ck)
        w = w.permute(0, 2, 1)
        w = torch.softmax(w, dim = -1)
        w = w.permute(0, 2, 1)
        w = w.view(N, pt1, pt2, Ck)
        x = torch.sum(w * x_v, dim = 2)

        hi = x.unsqueeze(2).expand(-1, -1, n_pts, Ck) 
        hj = x.unsqueeze(1).expand(-1, n_pts, -1, Ck) 
        mask = (p[:, :, 0] == 0.).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_pts, 1)
        pij = pi - pj
        ft_mask = mask.expand(-1, -1, n_pts, self.out_planes)
        norm =  (pij [..., 3]**2 - torch.sum(pij [..., 0:3] **2, dim = -1)).unsqueeze(-1).masked_fill(mask, 0)
        M = registered_mask.unsqueeze(1).expand(-1, n_pts, 4)
        mij = self.phi_e(torch.cat([hi, hj, norm], dim = -1)).masked_fill(ft_mask, -1e4) 
        propagate_weight = torch.softmax(self.phi_x(mij), dim = 2)
        new_p = (p + (1./M)*torch.sum(propagate_weight*pij, dim = 2)).masked_fill(p == 0., 0) 
        if self.return_attn:
            return x, new_p, w
        return x, new_p

@torch.jit.script
def to_ptrapphie(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Conver px, py, pz, E to pt, eta, phi, e coordinate.
    Args:
        p (torch.Tensor): Tensor of shape (N, 4), representing the particle's four momentum 
    Return:
        (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): 
        A tuple containing:
            - pt (torch.Tensor): Tranverse momentum
            - rapidity (torch.Tensor): Rapidity along the beam axis
            - phi (torch.Tensor): Azimuthal angle in the tranverse plane.
            - energy (torch.Tensor): Energy, unchanged
    '''
    
    px, py, pz, energy = p.split((1, 1, 1, 1), dim = -1)
    
    pt = torch.sqrt(px * px + py * py)
    
    rapidity = 0.5 * torch.log(1 + (2 * pz)/ (energy - pz ).clamp(min = 1e-20))
    phi = torch.atan2(py, px)
    
    return pt, rapidity, phi, energy

@torch.jit.script
def delta_phi(phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    '''
    Compute delta phi between two particles.
    
    Args:
        phi1 (torch.Tensor): prepare phi of first particle
        phi2 (torch.Tensor): prepare phi of second particle
    
    ''' 
    return ((phi2 - phi1) + torch.pi) % (2 * torch.pi) - torch.pi 

@torch.jit.script
def prepare_interaction(xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
    """
    Prepare pairwise interaction features between particles based on their four-momenta.
    
    Given two sets of particle four-momenta, computes physically-motivated features:
    - DeltaR (rapidity-phi distance)
    - Transverse momentum imbalance (kt)
    - energy fraction (z)
    - Invariance mass (m^2)
    
    Args:
        xi (torch.Tensor): Tensor of shape (N, 4), representing the first particle's four momentum
        xj (torch.Tensor): Tensor of shape (N, 4), representing the second particle's four momentum

    
    """ 
    pt_i, rapidity_i, phi_i, _ = to_ptrapphie(xi)
    pt_j, rapidity_j, phi_j, _ = to_ptrapphie(xj) 
    
    delta = torch.sqrt((rapidity_i - rapidity_j) ** 2 + delta_phi(phi_i, phi_j) ** 2).clamp(min = 1e-20) 
    ptmin = torch.minimum(pt_i, pt_j)
    
    kt = (ptmin * delta).clamp(min = 1e-8)
    z = (ptmin / (pt_i + pt_j).clamp(min = 1e-8)).clamp(min = 1e-8) 

    xij = xi + xj
    m2 = (xij[..., 3]**2 - torch.sum(xij[..., 0:3] **2, dim = -1)).unsqueeze(-1).clamp(min = 1e-8)
    
    return torch.log(torch.cat((delta, kt, z, m2), dim = -1))
    