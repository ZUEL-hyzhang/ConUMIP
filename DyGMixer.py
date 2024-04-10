import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder,DishTS
from utils.utils import NeighborSampler


class DyGMixer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu'):
        """
        DyGFormer model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(DyGMixer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim,parameter_requires_grad=False)
        
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        # self.projection_layer = nn.Linear(in_features=self.node_feat_dim+self.edge_feat_dim+self.time_feat_dim+self.neighbor_co_occurrence_feat_dim, out_features=self.edge_feat_dim, bias=True)

        # self.projection_layer = nn.Linear(in_features = self.node_feat_dim+self.edge_feat_dim+self.time_feat_dim, out_features=self.edge_feat_dim, bias=True)
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.edge_feat_dim, bias=True)
        })
        # self.projection_layer = nn.ModuleDict({
        #     'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
        #     'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
        #     'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
        #     'neighbor_co_occurrence': nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        # })
        
     
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=max_input_sequence_length, num_channels=self.edge_feat_dim,
                     token_dim_expansion_factor=0.5,
                     channel_dim_expansion_factor=4, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
       

        # self.transformers = nn.ModuleList([
        #     TransformerEncoder(seq_len=max_input_sequence_length,attention_dim=self.edge_feat_dim, num_heads=self.num_heads, dropout=self.dropout)
        #     for _ in range(self.num_layers)
        # ])
       
        # self.complex_weight = nn.Parameter(torch.randn(max_input_sequence_length//2+1, self.edge_feat_dim , 2,dtype=torch.float32))
        self.pred = nn.Linear(max_input_sequence_length,1)
        # self.output_layer = nn.Linear(in_features=self.edge_feat_dim, out_features=self.edge_feat_dim, bias=True)
        # self.dishts=DishTS(dish_init='uniform',n_series=self.edge_feat_dim,seq_len=max_input_sequence_length)
      

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)
        
        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)
             # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
        # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        
        src_padded_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_padded_nodes_neighbor_node_raw_features)
        src_padded_nodes_edge_raw_features = self.projection_layer['edge'](src_padded_nodes_edge_raw_features)
        src_padded_nodes_neighbor_time_features = self.projection_layer['time'](src_padded_nodes_neighbor_time_features)
        src_padded_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_padded_nodes_neighbor_co_occurrence_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_padded_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_padded_nodes_neighbor_node_raw_features)
        dst_padded_nodes_edge_raw_features = self.projection_layer['edge'](dst_padded_nodes_edge_raw_features)
        dst_padded_nodes_neighbor_time_features = self.projection_layer['time'](dst_padded_nodes_neighbor_time_features)
        dst_padded_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_padded_nodes_neighbor_co_occurrence_features)

        # src_combined_features = torch.cat([src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features], dim=-1)
        # dst_combined_features = torch.cat([dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features], dim=-1)
        src_combined_features = src_padded_nodes_neighbor_node_raw_features+ src_padded_nodes_edge_raw_features+ src_padded_nodes_neighbor_time_features+src_padded_nodes_neighbor_co_occurrence_features
        dst_combined_features = dst_padded_nodes_neighbor_node_raw_features+ dst_padded_nodes_edge_raw_features+ dst_padded_nodes_neighbor_time_features+dst_padded_nodes_neighbor_co_occurrence_features

        # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_edge_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * edge_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        # src_combined_features = self.projection_layer(src_combined_features)
        # dst_combined_features = self.projection_layer(dst_combined_features)
        # src = torch.fft.rfft(src_combined_features, n=self.max_input_sequence_length, dim=1, norm='forward')
        # # origin_x =x.clone()
        # weight = torch.view_as_complex(self.complex_weight)
        # src=src*weight
        # src_combined_features = torch.fft.irfft(src, n=self.max_input_sequence_length, dim=1, norm='forward')

        # dst = torch.fft.rfft(dst_combined_features, n=self.max_input_sequence_length, dim=1, norm='forward')
        # # origin_x =x.clone()
        # weight = torch.view_as_complex(self.complex_weight)
        # dst=dst*weight
        # dst_combined_features = torch.fft.irfft(dst, n=self.max_input_sequence_length, dim=1, norm='forward')
        
        # src_combined_features,_=self.dishts(src_combined_features,'forward')
        # dst_combined_features,_=self.dishts(dst_combined_features,'forward')
        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            src_combined_features = mlp_mixer(input_tensor=src_combined_features)
        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            dst_combined_features = mlp_mixer(input_tensor=dst_combined_features)
        
        # for transformer in self.transformers:
        #     src_combined_features = transformer(src_combined_features)
        # for transformer in self.transformers:
        #     dst_combined_features = transformer(dst_combined_features)

        src_combined_features = self.pred(src_combined_features.permute(0,2,1)).squeeze(-1)
        dst_combined_features = self.pred(dst_combined_features.permute(0,2,1)).squeeze(-1)
       
        # src_combined_features=self.dishts(src_combined_features,'inverse')
        # dst_combined_features=self.dishts(dst_combined_features,'inverse')
       
       
        # src_combined_features = torch.mean(src_combined_features, dim=1)
        # # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        # dst_combined_features = torch.mean(dst_combined_features, dim=1)
       
      
        # event_combined_features=self.dishts(event_combined_features.unsqueeze(1),'inverse').squeeze(1)
        # Tensor, shape (batch_size, node_feat_dim)
        # src_node_embeddings = self.output_layer(src_combined_features)
        # # Tensor, shape (batch_size, node_feat_dim)
        # dst_node_embeddings = self.output_layer(dst_combined_features)

        # src_node_embeddings = src_combined_features

        # # Tensor, shape (batch_size, node_feat_dim)
        # dst_node_embeddings = dst_combined_features
       
        return src_combined_features, dst_combined_features

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 128):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size  == 0
        
        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_input_sequence_length)).astype(np.long)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_input_sequence_length)).astype(np.long)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_input_sequence_length)).astype(np.float32)
        
        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))
        

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0
    

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features
    
    def get_patches(self, combined_features: torch.Tensor, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert combined_features.shape[1] % patch_size == 0
        num_patches = combined_features.shape[1] // patch_size
       

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_combined_features= []
        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_combined_features.append(combined_features[:, start_idx: end_idx, :])
            
            
        # print(combined_features.shape)
        # print(patch_size)
        # print(patches_combined_features.shape)
        batch_size = len(combined_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_combined_features = torch.stack(patches_combined_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
       
        return patches_combined_features


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

class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # two lists to store the appearances of source and destination nodes
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids):

            # src_unique_keys, ndarray, shape (num_src_unique_keys, )
            # src_inverse_indices, ndarray, shape (src_max_seq_length, )
            # src_counts, ndarray, shape (num_src_unique_keys, )
            # we can use src_unique_keys[src_inverse_indices] to reconstruct the original input, and use src_counts[src_inverse_indices] to get counts of the original input
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the source node
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            # dst_unique_keys, ndarray, shape (num_dst_unique_keys, )
            # dst_inverse_indices, ndarray, shape (dst_max_seq_length, )
            # dst_counts, ndarray, shape (num_dst_unique_keys, )
            # we can use dst_unique_keys[dst_inverse_indices] to reconstruct the original input, and use dst_counts[dst_inverse_indices] to get counts of the original input
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the destination node
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (src_max_seq_length, 2)
            src_padded_nodes_appearances.append(torch.stack([src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))

            # we need to use copy() to avoid the modification of dst_padded_node_neighbor_ids
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
            # Tensor, shape (dst_max_seq_length, 2)
            dst_padded_nodes_appearances.append(torch.stack([dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], dim=1))

        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        # set the appearances of the padded node (with zero index) to zeros
        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids == 0)] = 0.0
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                                                                  dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        
        # Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)

        src_padded_nodes_neighbor_co_occurrence_features =  (src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_padded_nodes_neighbor_co_occurrence_features =  (dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)


        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        
        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features


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

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)
class FilterLayer(nn.Module):
    def __init__(self,max_input_length:int, hidden_dim: int):
        super(FilterLayer, self).__init__()
        self.max_input_length=max_input_length
        self.complex_weight = nn.Parameter(torch.randn(1,self.max_input_length//2+1, hidden_dim, 2,dtype=torch.float32))
        self.Dropout = nn.Dropout(0.5)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
    def forward(self, input_tensor: torch.Tensor):
      

        batch,seq_len, hidden = input_tensor.shape
        hidden_states=input_tensor
        low=int(np.around(seq_len/2/3))
        band=2*low

        # if filter_type=='adap':
        x = torch.fft.rfft(hidden_states, n=self.max_input_length, dim=1, norm='forward')
        # origin_x =x.clone()
        weight = torch.view_as_complex(self.complex_weight)
        x=x*weight
        sequence_emb_fft = torch.fft.irfft(x, n=self.max_input_length, dim=1, norm='forward')
        sequence_emb_fft = sequence_emb_fft[:,0:seq_len,:]
        hidden_states = self.Dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

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

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)
        self.filter1=FilterLayer(max_input_length=num_tokens,hidden_dim=num_channels)
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)
        # self.filter2=FilterLayer(max_input_length=num_tokens,hidden_dim=num_channels)

    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        hidden_tensor=self.filter1(input_tensor)
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(hidden_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor


class TransformerEncoder(nn.Module):

    def __init__(self, seq_len: int, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)
        self.filter=FilterLayer(max_input_length=seq_len,hidden_dim=attention_dim)
        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """

        inputs = self.filter(inputs)
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs


# class MLPMixer(nn.Module):

#     def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
#                  channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
#         """
#         MLP Mixer.
#         :param num_tokens: int, number of tokens
#         :param num_channels: int, number of channels
#         :param token_dim_expansion_factor: float, dimension expansion factor for tokens
#         :param channel_dim_expansion_factor: float, dimension expansion factor for channels
#         :param dropout: float, dropout rate
#         """
#         super(MLPMixer, self).__init__()

#         # self.token_norm = nn.LayerNorm(num_tokens)
#         # self.channel_norm = nn.LayerNorm(num_channels)
#         # self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
#         #                                         dropout=dropout)
#         self.prenorm = map_norm(norm_type='LN', embedding_size = num_channels)
#         self.norm = map_norm(norm_type='LN', embedding_size = num_channels)
#         self.f1 = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,
#                                                   dropout=dropout)
        
#         self.f2 = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,
#                                                   dropout=dropout)
    

#     def forward(self, input_tensor: torch.Tensor):
#         """
#         mlp mixer to compute over tokens and channels
#         :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
#         :return:
#         """
#         # mix tokens
#         # Tensor, shape (batch_size, num_channels, num_tokens)
#         # hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
#         # Tensor, shape (batch_size, num_tokens, num_channels)
#         # hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
#         input_tensor = self.prenorm(input_tensor)
#         hidden_tensor = self.f1(input_tensor)
#         # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
#         output_tensor = hidden_tensor + input_tensor
#         output_tensor = self.norm(output_tensor)
#         # mix channels
#         # Tensor, shape (batch_size, num_tokens, num_channels)
#         # hidden_tensor = self.channel_norm(output_tensor)
#         # Tensor, shape (batch_size, num_tokens, num_channels)
        
#         hidden_tensor = self.f2(hidden_tensor)
#         # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
#         output_tensor = hidden_tensor + output_tensor

#         return output_tensor

# class FeedForwardNet(nn.Module):

#     def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
#         """
#         two-layered MLP with GELU activation function.
#         :param input_dim: int, dimension of input
#         :param dim_expansion_factor: float, dimension expansion factor
#         :param dropout: float, dropout rate
#         """
#         super(FeedForwardNet, self).__init__()

#         self.input_dim = input_dim
#         self.dim_expansion_factor = dim_expansion_factor
#         self.dropout = dropout

#         self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
#                                  nn.GELU(),
#                                  nn.Dropout(dropout),
#                                  nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
#                                  nn.Dropout(dropout))
#         self.rotate = RotateChord(track_size=22)
#     def forward(self, x: torch.Tensor):
#         """
#         feed forward net forward process
#         :param x: Tensor, shape (*, input_dim)
#         :return:
#         """
#         x = self.ffn(x)
#         x = self.rotate(x, x.shape[1])
#         return x


# def map_norm(norm_type, embedding_size, track_size=None):
#     """
#     Maps the given normalization type to the corresponding PyTorch module.

#     Args:
#         norm_type (str): The normalization type ('LN', 'BN', 'GN', or None).
#         embedding_size (int): The size of the token embeddings.
#         track_size (int, optional): The number of groups for Group Normalization.

#     Returns:
#         nn.Module: The corresponding normalization module.
#     """
#     if norm_type == 'LN':
#         norm = nn.LayerNorm(embedding_size)
#     elif norm_type == 'BN':
#         norm = BatchNorm(embedding_size)
#     elif norm_type == 'GN':
#         norm = GroupNorm(embedding_size, track_size)
#     elif norm_type == 'None':
#         norm = nn.Identity()
#     return norm

    
# class RotateChordVarLen(nn.Module):
#     """
#     A PyTorch module that performs a parameter-free rotation of tracks within variable-length token embeddings.

#     This module can be used to augment or modify the input data in a data-driven manner. The rotation is
#     performed separately for all sequences in a batch and is based on powers of 2. This version is designed to
#     handle variable-length input sequences of extremely diverse range.
    
#     No padding is applied.

#     Args:
#         track_size (int): The size of tracks to be rotated.
#     """
#     def __init__(self, track_size):
#         super().__init__()
#         self.track_size = track_size

#     def forward(self, x, lengths):
#         ys = torch.split(
#             tensor=x,
#             split_size_or_sections=lengths.tolist(),
#             dim=0
#         )

#         zs = []

#         # Roll sequences separately
#         for y in ys:
#             y = torch.split(
#                 tensor=y,
#                 split_size_or_sections=self.track_size,
#                 dim=-1
#             )
#             z = [y[0]]
#             for i in range(1, len(y)):
#                 offset = -2 ** (i - 1)
#                 z.append(torch.roll(y[i], shifts=offset, dims=0))
#             z = torch.cat(z, -1)
#             zs.append(z)

#         z = torch.cat(zs, 0)
        
#         return z

# class RotateChord(nn.Module):
#     """
#     A PyTorch module that performs a parameter-free rotation of tracks within token embeddings.

#     This module can be used to augment or modify the input data in a data-driven manner. The rotation is
#     performed jointly for all sequences in a batch and is based on powers of 2 (Chord protocol).

#     Args:
#         track_size (int): The size of tracks to be rotated.
#     """
#     def __init__(self, track_size):
#         super().__init__()
#         self.track_size = track_size

#     def forward(self, x, lengths=None):
#         y = torch.split(
#             tensor=x,
#             split_size_or_sections=self.track_size,
#             dim=-1
#         )
#         # Roll sequences in a batch jointly
#         # The first track remains unchanged
#         z = [y[0]]

#         for i in range(1, len(y)):
#             offset = - 2 ** (i - 1)
#             z.append(torch.roll(y[i], shifts=offset, dims=1))

#         z = torch.cat(z, -1)
      
#         return z