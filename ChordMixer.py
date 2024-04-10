import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder,DishTS
from utils.utils import NeighborSampler

    




class ChordMixer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_tokens: int, num_layers: int = 2, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.1, device: str = 'cpu'):
        """
        TCL model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_tokens: int, number of tokens
        :param num_layers: int, number of transformer layers
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(ChordMixer, self).__init__()
        
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.token_dim_expansion_factor = token_dim_expansion_factor
        self.channel_dim_expansion_factor = channel_dim_expansion_factor
        self.dropout = dropout
        self.device = device

        # self.complex_weight = nn.Parameter(torch.randn(20//2+1,self.edge_feat_dim + self.time_feat_dim , 2,dtype=torch.float32))
        self.num_channels = self.edge_feat_dim
        # in GraphMixer, the time encoding function is not trainable
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.projection_layer = nn.Linear(self.edge_feat_dim + time_feat_dim, self.num_channels)

        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_tokens, num_channels=self.num_channels,
                     token_dim_expansion_factor=self.token_dim_expansion_factor,
                     channel_dim_expansion_factor=self.channel_dim_expansion_factor, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels + self.node_feat_dim, out_features=self.node_feat_dim, bias=True)
        self.dishts=DishTS(dish_init='standard',n_series=self.num_channels,seq_len=20)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20, time_gap: int = 2000):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors, time_gap=time_gap)

        return src_node_embeddings, dst_node_embeddings

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         num_neighbors: int = 20, time_gap: int = 2000):
        """
        given node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings of nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :param time_gap: int, time gap for neighbors to compute node features
        :return:
        """
        # link encoder
        # get temporal neighbors, including neighbor ids, edge ids and time information
        # neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # neighbor_times, ndarray, shape (batch_size, num_neighbors)
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        nodes_neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_times).float().to(self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        nodes_neighbor_time_features[torch.from_numpy(neighbor_node_ids == 0)] = 0.0

        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim + time_feat_dim)
        combined_features = torch.cat([nodes_edge_raw_features, nodes_neighbor_time_features], dim=-1)
        # filter
        # x = torch.fft.rfft(combined_features, n=num_neighbors, dim=1, norm='forward')
        # # origin_x =x.clone()
        # weight = torch.view_as_complex(self.complex_weight)
        # x=x*weight
        # combined_features_filter = torch.fft.irfft(x, n=num_neighbors, dim=1, norm='forward')
        # combined_features_filter = self.out_dropout(combined_features_filter)
        # combined_features = self.LayerNorm(combined_features + combined_features_filter)
        # combined_features=combined_features+combined_features_filter
        # combined_features = sequence_emb_fft[0:num_neighbors,:]

        # Tensor, shape (batch_size, num_neighbors, num_channels)
        combined_features = self.projection_layer(combined_features)
        combined_features,_=self.dishts(combined_features,'forward')
    

        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            combined_features = mlp_mixer(input_tensor=combined_features)
        
        combined_features=self.dishts(combined_features,'inverse')
        # Tensor, shape (batch_size, num_channels)
        combined_features = torch.mean(combined_features, dim=1)

        # node encoder
        # get temporal neighbors of nodes, including neighbor ids
        # time_gap_neighbor_node_ids, ndarray, shape (batch_size, time_gap)
        time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                                                          node_interact_times=node_interact_times,
                                                                                          num_neighbors=time_gap)

        # Tensor, shape (batch_size, time_gap, node_feat_dim)
        nodes_time_gap_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(time_gap_neighbor_node_ids)]

        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask = torch.from_numpy((time_gap_neighbor_node_ids > 0).astype(np.float32))
        # note that if a node has no valid neighbor (whose valid_time_gap_neighbor_node_ids_mask are all zero), directly set the mask to -np.inf will make the
        # scores after softmax be nan. Therefore, we choose a very large negative number (-1e10) instead of -np.inf to tackle this case
        # Tensor, shape (batch_size, time_gap)
        valid_time_gap_neighbor_node_ids_mask[valid_time_gap_neighbor_node_ids_mask == 0] = -1e10
        # Tensor, shape (batch_size, time_gap)
        scores = torch.softmax(valid_time_gap_neighbor_node_ids_mask, dim=1).to(self.device)

        # Tensor, shape (batch_size, node_feat_dim), average over the time_gap neighbors
        nodes_time_gap_neighbor_node_agg_features = torch.mean(nodes_time_gap_neighbor_node_raw_features * scores.unsqueeze(dim=-1), dim=1)

        # Tensor, shape (batch_size, node_feat_dim), add features of nodes in node_ids
        output_node_features = nodes_time_gap_neighbor_node_agg_features + self.node_raw_features[torch.from_numpy(node_ids)]

        # Tensor, shape (batch_size, node_feat_dim)
        node_embeddings = self.output_layer(torch.cat([combined_features, output_node_features], dim=1))

        return node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        # self.token_norm = nn.LayerNorm(num_tokens)
        # self.channel_norm = nn.LayerNorm(num_channels)
        # self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
        #                                         dropout=dropout)
        self.prenorm = map_norm(norm_type='LN', embedding_size = num_channels)
        self.norm = map_norm(norm_type='LN', embedding_size = num_channels)
        self.f1 = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)
        
        self.f2 = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)
    

    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        # hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        # hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        input_tensor = self.prenorm(input_tensor)
        hidden_tensor = self.f1(input_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor
        output_tensor = self.norm(output_tensor)
        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        # hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        
        hidden_tensor = self.f2(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor

class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))
        self.rotate = RotateChord(track_size=34)
    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        x = self.ffn(x)
        x = self.rotate(x, x.shape[1])
        return x


def map_norm(norm_type, embedding_size, track_size=None):
    """
    Maps the given normalization type to the corresponding PyTorch module.

    Args:
        norm_type (str): The normalization type ('LN', 'BN', 'GN', or None).
        embedding_size (int): The size of the token embeddings.
        track_size (int, optional): The number of groups for Group Normalization.

    Returns:
        nn.Module: The corresponding normalization module.
    """
    if norm_type == 'LN':
        norm = nn.LayerNorm(embedding_size)
    elif norm_type == 'BN':
        norm = BatchNorm(embedding_size)
    elif norm_type == 'GN':
        norm = GroupNorm(embedding_size, track_size)
    elif norm_type == 'None':
        norm = nn.Identity()
    return norm

    
class RotateChordVarLen(nn.Module):
    """
    A PyTorch module that performs a parameter-free rotation of tracks within variable-length token embeddings.

    This module can be used to augment or modify the input data in a data-driven manner. The rotation is
    performed separately for all sequences in a batch and is based on powers of 2. This version is designed to
    handle variable-length input sequences of extremely diverse range.
    
    No padding is applied.

    Args:
        track_size (int): The size of tracks to be rotated.
    """
    def __init__(self, track_size):
        super().__init__()
        self.track_size = track_size

    def forward(self, x, lengths):
        ys = torch.split(
            tensor=x,
            split_size_or_sections=lengths.tolist(),
            dim=0
        )

        zs = []

        # Roll sequences separately
        for y in ys:
            y = torch.split(
                tensor=y,
                split_size_or_sections=self.track_size,
                dim=-1
            )
            z = [y[0]]
            for i in range(1, len(y)):
                offset = -2 ** (i - 1)
                z.append(torch.roll(y[i], shifts=offset, dims=0))
            z = torch.cat(z, -1)
            zs.append(z)

        z = torch.cat(zs, 0)
        return z

class RotateChord(nn.Module):
    """
    A PyTorch module that performs a parameter-free rotation of tracks within token embeddings.

    This module can be used to augment or modify the input data in a data-driven manner. The rotation is
    performed jointly for all sequences in a batch and is based on powers of 2 (Chord protocol).

    Args:
        track_size (int): The size of tracks to be rotated.
    """
    def __init__(self, track_size):
        super().__init__()
        self.track_size = track_size

    def forward(self, x, lengths=None):
        y = torch.split(
            tensor=x,
            split_size_or_sections=self.track_size,
            dim=-1
        )
        # Roll sequences in a batch jointly
        # The first track remains unchanged
        z = [y[0]]

        for i in range(1, len(y)):
            offset = - 2 ** (i - 1)
            z.append(torch.roll(y[i], shifts=offset, dims=1))

        z = torch.cat(z, -1)
      
        return z