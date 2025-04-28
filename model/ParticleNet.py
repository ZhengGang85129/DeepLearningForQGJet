import torch
from torch import nn
from collections import OrderedDict
from typing import Tuple, List, Dict

@torch.jit.script
def point_distance_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r_A = torch.unsqueeze(torch.sum(A*A, dim=2), dim=2)
    r_B = torch.unsqueeze(torch.sum(B*B, dim=2), dim=2)
    A_dot_B = torch.matmul(A, torch.permute(B, (0, 2, 1)))

    return r_A - 2 * A_dot_B + torch.permute(r_B, (0, 2, 1))

@torch.jit.script
def gather_nd(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    '''
        tensor : (N, P, C)
        index  : (N, P, K)
        output : (N, P, K, C)
    '''
    N = index.shape[0]
    P = index.shape[1]
    K = index.shape[2]
    C = tensor.shape[2]

    index = index.reshape(N, P*K) #(N, P*K)

    return torch.cat( [tensor[i,sub_index] for i, sub_index in enumerate(index)] ).reshape(N, P, K, C)

@torch.jit.script
def knn(points: torch.Tensor, K: int = 2):

    #Calculate the distance of all points
    distance = point_distance_matrix(points, points)

    #Find out the indices of k nearest neighbor
    _, topk_index = torch.topk(-distance, k=K+1)
    return topk_index[:,:,1:]

@torch.jit.script
def feature_redefine(features: torch.Tensor, knn_index: torch.Tensor):
    """
    Args:
        features: (N , P , C)
        knn_index: (N , P , K)
    Return:
        Tensor of center and neighbor features concatenation. dim: (N, P, K, 2 * C)
    """
    f  = gather_nd(features, knn_index) #(N, P, K, C)
    fc = torch.tile(torch.unsqueeze(features, dim=2), (1,1,f.shape[2],1)) #(N, P, K, C)
    ff = torch.cat([fc, torch.subtract(fc, f)], dim=-1) #(N, P, K, 2*C)

    return torch.permute(ff, (0, 3, 1, 2))

class Edge_Conv(nn.Module):
    """
    args:
        index: Edge_Conv layer index
        edge_conv_parameters:
            List of edge-convolution parameters
    """
    def __init__(self, index: int, edge_conv_parameters: Tuple[int, List[int]]):
        super().__init__()

        self.index = index
        self.K, self.channel_list = edge_conv_parameters

        self.conv_layer = self._make_conv_layer()
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.channel_list[0][0], self.channel_list[-1][-1], kernel_size=(1,1)),
            nn.BatchNorm2d(self.channel_list[-1][-1])
        )
        self.final_act = nn.ReLU()

    # Convolution layer maker
    def _make_conv_layer(self):

        '''
            [Conv2d--BatchNorm2d--ReLU] * n
        '''

        layer = []

        for i_conv, (C_in, C_out) in enumerate(self.channel_list):

            layer.append( ('edge_conv_Conv2d_{}_{}'.format(self.index, i_conv), nn.Conv2d(C_in*2 if i_conv == 0 else C_in, C_out, kernel_size=(1,1))) )
            layer.append( ('edge_conv_BatchNorm2d_{}_{}'.format(self.index, i_conv), nn.BatchNorm2d(C_out)) )
            layer.append( ('edge_conv_ReLU_{}_{}'.format(self.index, i_conv), nn.ReLU()) ) #(N, C, P, K)

        return nn.Sequential( OrderedDict(layer) )


    def forward(self, features):

        '''
            points  : (N, P, C)
            feature : (N, P, C) if index == 0
                      (N, C, P) if index > 0
        '''

        # If second edge convolution , permutation (N, C, P) ---> (N, P, C)
        if self.index != 0:
            features = torch.permute(features, (0,2,1))

        X = features

        # The first edge convolution chooses (eta, phi) to judge the point distance and the others as input features
        pts = X[:,:,0:2] if self.index == 0 else X
        X = X[:,:,2:] if self.index == 0 else X
        X_shortcut = X

        # knn method
        knn_index = knn(pts, K=self.K) #(N, P, K)
        X = feature_redefine(X, knn_index) #(N, 2*C, P, K), 2 means x and x'

        # Convolution layer
        X = self.conv_layer(X) #(N, C', P, K)

        # Aggregation
        X = torch.mean(X, dim=3) #(N, C', P)

        # Residual 
        init_X = torch.unsqueeze(torch.permute(X_shortcut, (0,2,1)), dim=3) #(N, C, P, K=1)
        init_X = self.shortcut(init_X) #(N, C', P, K=1)
        init_X = torch.squeeze(init_X, dim=3) #(N, C', P)

        return self.final_act(X+init_X) #(N, C', P)

class ParticleNet(nn.Module):

    def __init__(self, particle_net_parameters: Dict, isFinalModule=True):
        super().__init__()

        self.isFinalModule = isFinalModule

        self.edge_conv_parameters = particle_net_parameters['edge_conv']
        self.fc_parameters = particle_net_parameters['fc']

        self.input_BatchNorm2d = nn.BatchNorm2d(self.edge_conv_parameters[0][1][0][0])

        self.Edge_Conv   = self._make_edge_conv()
        self.FC          = self._make_fully_connected_layer()
        self.final_layer = nn.Linear(self.fc_parameters[-1][-1][-1], 2)


    def _make_edge_conv(self):

        block = []

        for i_block, param in enumerate(self.edge_conv_parameters):
            block.append( ('edge_conv_{}'.format(i_block), Edge_Conv(i_block, param)) )

        return nn.Sequential( OrderedDict(block) )


    def _make_fully_connected_layer(self):

        layer = []

        for i_layer, param in enumerate(self.fc_parameters):

            drop_rate, nodes = param

            layer.append( ('Linear_{}'.format(i_layer), nn.Linear(nodes[0], nodes[1])) )
            layer.append( ('ReLU_{}'.format(i_layer), nn.ReLU()) )
            layer.append( ('Dropout_{}'.format(i_layer), nn.Dropout(p=drop_rate)) )

        return nn.Sequential( OrderedDict(layer) )


    def forward(self, features: torch.Tensor):

        '''
            features : (N, P, C)
        '''

        # Extract (eta, phi) information from feature to avoid the first batch normalization
        points = features[:,:,0:2]
        # First batch normalization
        fts = torch.unsqueeze(torch.permute(features[:,:,2:], (0,2,1)), dim=3) # (N, C, P, K=1)
        fts = torch.permute(torch.squeeze(self.input_BatchNorm2d(fts), dim=3), (0,2,1)) # (N, P, C)

        # Conbination
        fts = torch.cat((points, fts), dim=2)

        # Edge convolution
        fts = self.Edge_Conv(fts) # (N, C', P)

        # Global average pooling
        fts = torch.mean(fts, dim=2) # (N, C')

        # Fully connected layer
        fts = self.FC(fts) # (N, C'')

        # Let Particle Net can be front module
        if not self.isFinalModule:
            return fts
        else:
            fts = self.final_layer(fts) # 2 classes

            return fts
