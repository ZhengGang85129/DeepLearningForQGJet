import torch
from torch import nn
class LGEB(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout = 0., c_weight=1.0, last_layer=False):
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2 # dims for Minkowski norm & inner product

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias = False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU())

        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output))

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi_x = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            layer)

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid())
        
        self.last_layer = last_layer
        if last_layer:
            del self.phi_x

    def m_model(self, hi, hj, norms, dots):
        out = torch.cat([hi, hj, norms, dots], dim=1)
        out = self.phi_e(out)
        w = self.phi_m(out)
        out = out * w
        return out

    def h_model(self, h, edges, m, node_attr):
        i, j = edges
        agg = unsorted_segment_sum(m, i, num_segments=h.size(0))
        agg = torch.cat([h, agg, node_attr], dim=1)
        out = h + self.phi_h(agg)
        return out

    def x_model(self, x, edges, x_diff, m):
        i, j = edges
        trans = x_diff * self.phi_x(m)
        # From https://github.com/vgsatorras/egnn
        # This is never activated but just in case it explosed it may save the train
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, i, num_segments=x.size(0))
        x = x + agg * self.c_weight
        return x

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = normsq4(x_diff).unsqueeze(1)
        dots = dotsq4(x[i], x[j]).unsqueeze(1)
        norms, dots = psi(norms), psi(dots)
        return norms, dots, x_diff

    def forward(self, h, x, edges, node_attr=None):
        i, j = edges
        norms, dots, x_diff = self.minkowski_feats(edges, x)

        m = self.m_model(h[i], h[j], norms, dots) # [B*N, hidden]
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m


def unsorted_segment_sum(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def normsq4(p):
    r''' Minkowski square norm
         `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
    ''' 
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)
    
def dotsq4(p,q):
    r''' Minkowski inner product
         `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
    '''
    psq = p*q
    return 2 * psq[..., 0] - psq.sum(dim=-1)
    
def psi(p):
    ''' `\psi(p) = Sgn(p) \cdot \log(|p| + 1)`
    '''
    return torch.sign(p) * torch.log(torch.abs(p) + 1)



class LGEB_adapt(nn.Module):
    def __init__(self, 
                 n_input:int , 
                 n_output:int,
                 n_hidden: int ,
                 c_weight:float = 1.0e-3,
                 n_scalar: int = 13,
                 last_layer = False
                 ) -> None:
        super(LGEB_adapt, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2 # dims for Minkowski norm & inner product
        self.last_layer = last_layer 
        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias = False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()  
        )
        
        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden* 2 + n_scalar, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output))
        
        layer = nn.Linear(n_hidden, 1, bias = False)
        torch.nn.init.xavier_uniform_(layer.weight, gain = 0.001)
        
        self.phi_x = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            layer
        )
        
        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        ) 
    
    def minkowski_feats(self, pi, pj):
        
        p_diff = pi - pj
        
        p_substract_q = torch.pow(p_diff, 2)
        norm_Feat = p_substract_q.sum(dim = -1) - 2 * p_substract_q[..., 0] 
        
        p_prod_q = pi * pj
        iprod_Feat = 2 * p_prod_q[..., 0] - p_prod_q.sum(dim = -1)
        
        norm_Feat = psi(norm_Feat).unsqueeze(-1)
        iprod_Feat = psi(iprod_Feat).unsqueeze(-1)
        return norm_Feat, iprod_Feat, p_diff 
        
    def m_model(self, hi, hj, norms, dots):
        
        feat_ij = torch.cat([hi, hj, norms, dots], dim = -1)
        
        batch_size, n_pts, emb_dim = feat_ij.shape[0], feat_ij.shape[1], feat_ij.shape[-1]
        feat_ij = feat_ij.view(-1, emb_dim)
        m_ij = self.phi_e(feat_ij) # 
        w_ij = self.phi_m(m_ij)
        out = m_ij * w_ij
        out = out.view(batch_size, n_pts, n_pts, -1)
        return out 
    def x_model(self, p, p_diff, wm_ij) -> torch.FloatTensor: 
        aggs = p_diff * self.phi_x(wm_ij)
        aggs = torch.clamp(aggs, min = -100, max = 100)
        aggs = torch.mean(aggs, dim = 2)
        p = p + aggs * self.c_weight        
        
        return p 
    
    def h_model(self, h, m, s) -> torch.FloatTensor:
        #raise ValueError(s.shape, h.shape, m.shape)
        aggs = torch.sum(m, dim = 2)    
        aggs = torch.cat([h, aggs, s], dim = -1)
        batch_size, n_pts = aggs.shape[0], aggs.shape[1]
        aggs = aggs.view(batch_size * n_pts, -1)
        phi_h = self.phi_h(aggs)
        phi_h = phi_h.view(batch_size, n_pts, -1)
        
    
        out = h + phi_h
        return out 
    
    def forward(self, hps) -> torch.FloatTensor:
        
        h, p, s = hps #
        
        n_pts = p.shape[1]
        pi = p.unsqueeze(2).expand(-1, -1, n_pts, -1)
        pj = p.unsqueeze(1).expand(-1, n_pts, -1, -1)

        norms, dots, p_diff = self.minkowski_feats(pi, pj)
        #norms -> (N, n, n, 1), (N, n, n, 1), (N, n, n, 4)
        hi = h.unsqueeze(2).expand(-1, -1, n_pts, -1)
        hj = h.unsqueeze(1).expand(-1, n_pts, -1, -1)
        
        wm_ij = self.m_model(hi = hi, hj = hj, norms = norms, dots = dots) 
        
        if not self.last_layer:
            p = self.x_model(p, p_diff, wm_ij) 
         
        h = self.h_model(h, m = wm_ij, s = s)
        return [h, p]
        
        
         
        
        
        
        
        
        
        
    
    
    


