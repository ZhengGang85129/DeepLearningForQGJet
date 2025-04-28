"""
MomentumCloudNet made by Zheng-Gang Chen
"""

import torch
import torch.nn as nn

from typing import List, Dict, Tuple, Union
import os, sys
sys.path.append(os.getcwd())

from JetTagger.utils.MomentumFormerLayer import MomentumFormerLayer 
from JetTagger.utils.MomentumFormerLayer import prepare_interaction


class MomentumCloudNetConfig(NamedTuple):
    embed_dims: List 
    blocks: List
    n_head: int 
    ksample: List
    stride: List


class ParticleFeatureEmbedding(nn.Module):
    '''
    A feature embedding module for particle cloud data, applying BatchNorm and multiple stacked layers of LayerNorm, Linear, and GELU activations.
    
    Args:
        input_dim (int): Dimension of input particle features.
        embedding_dims (List[int]): A list of integer specify the output dimensions for each embedding stage.
    '''
    def __init__(self, input_dim: int, embedding_dims: List[int]) -> None:
        super().__init__()
        self.BatchNorm1d = nn.BatchNorm1d(input_dim)
        embedding_chain = []
        _input_dim = input_dim
        for _embedding_dim in embedding_dims:
            embedding_chain.extend([
                nn.LayerNorm(_input_dim),
                nn.Linear(_input_dim, _embedding_dim),
                nn.GELU()
            ])
            _input_dim = _embedding_dim

        self.embedding = nn.Sequential(*embedding_chain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input x : [N, L, C]
        # Output x : [N, L, C']
        x = x.permute(0, 2, 1).contiguous() # [N, C, L]
        x = self.BatchNorm1d(x) 
        x = x.permute(0, 2, 1).contiguous()# [N, L, C]

        return self.embedding(x)

class Bottleneck(nn.Module):
    """
    Residual bottleneck block for particle cloud encoding.
    Integrating a point-based transformer layer with feature normalization and projection.
    Args:
        in_planes(int): Number of input feature channels.
        planes(int): Number of output feature channels.
        share_planes (int, optional): Number of neighboring points sampled for local aggregation.
        mode (str, optional): Interaction mode used in PoinTransformerLayer. e.g ('Int')
        return_attn(bool, optional): If True, returns attention weights from the transformer layer. Default is False.
    
    """
    
    def __init__(self, in_planes: int, planes: int, share_planes=8, nsample=16, mode = 'Int', return_attn: bool = False):
        super(Bottleneck, self).__init__()
        self.ln1 = nn.LayerNorm(in_planes)
        self.transformer = MomentumFormerLayer(planes, planes, share_planes = share_planes, nsample = nsample, mode = mode, return_attn = return_attn)
        self.return_attn = return_attn
        self.ln2 = nn.LayerNorm(planes)
        self.linear3 = nn.Linear(planes, planes, bias=False)
        self.ln3 = nn.LayerNorm(planes)
        self.gelu = nn.GELU()
        
    def forward(self, input: Dict):
        """
        Forward pass for the bottleneck block.
        Args:
         input(Dict): A dictionary containing:
          - x (Tensor): Particle feature tensor of shape (N, P, C_in).
          - p (Tensor): Particle momentum tensor of shape (N, P, 4).
          - U (Tensor): Interaction features of shape (N, P, P, F)
        Returns:
            Tuple:
                - y (Tensor): Updated particle features after residual connection. Shape (N, P, C_out)
                - new_p (Tensor): Updated particle momentum. Shape (N, P, 4).
                - attent_weight(Tensor): Attention weights, returned if `return_attn` is True.
        
        """
        
        x, p, U = input
        x = self.ln1(x)
        identity = x
        input = [x, p, U]
        if self.return_attn:
            y, new_p, attent_weight = self.transformer(input)
        else:
            y, new_p = self.transformer(input)
        y = self.gelu(self.ln2(y))
        y = self.ln3(self.linear3(y))
        y += identity
        y = self.gelu(y)
        if self.return_attn:
            return y, new_p, attent_weight
        return y, new_p


class MomentumCloudNetBase(nn.Module):
    '''
    A modular particle cloud classification network that integrates vector-based feature embedding, 
    momentum interaction updating, and multi-stage encoding with attention extraction.
    
    Args:
        block(nn.Module): The block module used for each encoder stage, typically our customized bottleneck(`Bottleneck`).
        blocks(List[int]): A list defining the number of output channels at each encoder stage.
        in_channels (int): The number of input feature channels for each particle. Default is 13.
        num_classes(int): The number of target classes for classification. Default is 2.
        ft_embeddims(List[int]): A list of integers specifying the embedding dimensions at feature embedding layers. 
        mode(str): Mode configuration passed to each encoder block. Default is an empty
    Attributes:
        ft_embed (nn.Module): Feature embedding module that projects raw particle feature into higher-dimensional spaces.
        enc (nn.ModuleList): A list of encoder blocks stacking progressively deeper feature representations.
        layerNorm (nn.LayerNorm): A final layer normalization applied before classification.
        cls (nn.Sequential): Final classification head mapping encoded features to class logits.
        Int_U (nn.Module): Linear layers that embed momentum-based interaction features.
    '''
    
    def __init__(self, block, blocks: List[int], in_channels=13, num_classes=2, ft_embeddims: List[int] = [64, 64], mode: None = '', ):
        super().__init__()
        self.in_planes = in_channels
        self.Layers = len(blocks)
        ft_embeddims.append(blocks[0])
        share_planes = 8
        self.ft_embed = self._make_ft_embed(ft_embeddims = ft_embeddims)
        self.enc = []
        for index in range(self.Layers):
            self.enc.append(self._make_enc(
            block,
            blocks[index],
            share_planes = share_planes,
            return_attn = False if index != self.Layers - 1 else True 
            ))
                        
        self.enc = nn.ModuleList(self.enc)
        
        self.layerNorm = nn.LayerNorm(blocks[-1]) 
        self.cls = nn.Sequential(
            nn.Linear(blocks[-1], blocks[-1]),
            nn.LayerNorm(blocks[-1]),
            nn.Linear(blocks[-1], num_classes),
        )
        self.Int_U = self.Lund_embed(4)
    def Lund_embed(self, n_input:int = 4) :
        linear_p = nn.Sequential(
                nn.Linear(n_input, 64, bias = False),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Linear(64, 128, bias = False),
                nn.GELU(),
            )
        return linear_p      
        
    def _make_ft_embed(self, ft_embeddims: List[int]):
        in_channels = self.in_planes
        self.in_planes = ft_embeddims[-1] 
        return ParticleFeatureEmbedding(input_dim = in_channels, embedding_dims = ft_embeddims)
    def _make_enc(self, block: Bottleneck, planes: int, share_planes: int = 8, mode: str = 'single', return_attn = False):
        layers = block(planes, planes, share_planes = share_planes, mode = mode, return_attn = return_attn)
        return layers

    def forward(self, xp: Tuple[torch.Tensor, torch.Tensor], return_attn: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Args:
            xp(Tuple[torch.Tensor, torch.Tensor]):
                A tuple containing:
                    - Particle features (Tensor): shape (N, P, C), where N = batch size, P = number pf particles, C = feature dimension.
                    - Particle momenta (Tensor): shape (N, P, 4), representing four-momentum (px, py, pz, E) or similar.
            return_attn (bool, optional): 
                If True, returns both classification output and the attention matrix from the final encoder stage. 
                If False, only returns the classification output. Default is True.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If `return_attn` is False: classification logits of shape (N, num_classes).
                - If `return_attn` is True: a tuple (classification logits, attention matrix).
        '''
        x, p = xp
        n_pts = p.shape[1] 
        pi = p.unsqueeze(2).expand(-1, -1, n_pts, -1)
        pj = p.unsqueeze(1).expand(-1, n_pts, -1, -1)
        U = self.Int_U(prepare_interaction(pi, pj))
        x = self.ft_embed(x)
        input = [x, p, U]
        
        for index, enc in enumerate(self.enc):
            if index != len(self.enc) - 1:
                x, p = enc(input = input)
            else:
                x, p, attention_matrix = enc(input = input)
            input = [x, p, U]
        y = torch.mean(x, dim = 1)
        out = self.cls(y)
        if return_attn:
            return out, attention_matrix
        return out
        
class MomentumCloudNet(MomentumCloudNetBase):
    def __init__(self, **kwargs):
        super(MomentumCloudNet, self).__init__(
            Bottleneck, **kwargs
        )
