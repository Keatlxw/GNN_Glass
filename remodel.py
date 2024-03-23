import torch
from torch import nn
from layer import GNNLayer
from utils import MLP

def global_max_pooling(node_features, batch):
    """
    全局最大池化函数。
    :param node_features: 节点特征矩阵，形状为 [num_nodes, num_features]
    :param batch: 一个向量，指示每个节点属于哪个图
    :return: 池化后的图级别特征
    """
    num_graphs = int(batch.max()) + 1
    max_pooled_features = []
    for graph_id in range(num_graphs):
        mask = (batch == graph_id)
        graph_features = node_features[mask]
        max_pooled_features.append(torch.max(graph_features, 0)[0])
    pooled_features = torch.stack(max_pooled_features, dim=0)
    return pooled_features

class GNNModel(nn.Module):
    def __init__(self, node_feature_len, edge_embedding_len,
                 init_node_embedding_units,
                 n_heads=(2,1,), attention_feature_lens=(30,15,),
                 task="classification", n_class=2, activation=None,
                 remember_func="residual",
                 in_dropout=0.1, attention_dropout=0.1, readout_dropout=0,
                 readout_hidden_units=None, device=None):
        super(GNNModel, self).__init__()
        assert task in ["classification", "regression"]
        self.pool = lambda x: torch.sum(x, dim=0)
        #self.pool = lambda x: torch.max(x, dim=0)[0]
        self.task = task
        self.n_heads = n_heads
        self.readout_hidden_units=readout_hidden_units
        self.remember_func = remember_func
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation = nn.Softplus() if activation is None else activation
        self.input_dropout_layer = nn.Dropout(in_dropout)
        self.judges=(False,True,)

        if init_node_embedding_units[-1] != node_feature_len:
            init_node_embedding_units = list(init_node_embedding_units[:-1]) + [node_feature_len]
        self.node_embedding_layer = MLP(node_feature_len, init_node_embedding_units, activation=self.activation)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(node_embedding_len=node_feature_len,
                     edge_embedding_len=edge_embedding_len, n_head=n_head,
                     attention_len=attention_len, attention_dropout=attention_dropout,
                     remember_func=remember_func, device=self.device,use_graph_level_attention=judge)
            for n_head, attention_len,judge in zip(n_heads, attention_feature_lens,self.judges)])

        if self.remember_func == "lstm":
            self.lstm_hidden = node_feature_len
            self.lstm = nn.LSTM(input_size=node_feature_len, hidden_size=self.lstm_hidden, batch_first=True)

        if readout_hidden_units is not None:
            self.readout_hidden_units = readout_hidden_units
            self.readout_hidden_layers = MLP(node_feature_len, readout_hidden_units, activation=self.activation)
            self.readout_layer = nn.Linear(readout_hidden_units[-1], n_class if task == "classification" else 1)
        else:
            self.readout_layer = nn.Linear(node_feature_len, n_class if task == "classification" else 1)
        self.readout_dropout_layer = nn.Dropout(readout_dropout)

        if task == "classification":
            self.logsoftmax = nn.LogSoftmax(dim=1)  # 修改为dim=1以适应批处理

    def forward(self, node_features, edge_features, neighbor_indices, neighbor_masks, mini_batch_hops_indices=None):
        #node_features = self.input_dropout_layer(self.node_embedding_layer(node_features))
        node_features = self.node_embedding_layer(self.input_dropout_layer(node_features))
        edge_features = edge_features
        neighbor_indices = neighbor_indices.type(torch.long)
        neighbor_masks = neighbor_masks
        graph_feature=0
        heads_attention_weights=0

        # 仅当 remember_func 为 "lstm" 时初始化 h 和 c
        if self.remember_func == "lstm":
            h = torch.zeros(1, node_features.size(0), self.lstm_hidden).to(self.device)
            c = torch.zeros(1, node_features.size(0), self.lstm_hidden).to(self.device)
        else:
            h, c = None, None  # 如果不使用 LSTM，将 h 和 c 设置为 None


        if mini_batch_hops_indices is None:
            for gnn_layer in self.gnn_layers:
                graph_feature,node_features, edge_features, (h, c),heads_attention_weights = gnn_layer(
                    node_features, edge_features, neighbor_indices,
                    neighbor_masks, h, c)
        else:
            # graph mini-batch process
            assert len(self.gnn_layers) == len(mini_batch_hops_indices) - 1
            for i, (gnn_layer, neighbor_idx_hop) in enumerate(zip(
                    self.gnn_layers, mini_batch_hops_indices)):
                if i != 0:
                    re_indices = torch.ones(len(node_features)).long() * - 1
                    map_indices = torch.arange(0, len(neighbor_idx_hop))
                    re_indices[neighbor_idx_hop] = map_indices
                    neighbor_indices = re_indices[neighbor_indices[neighbor_idx_hop]]

                    fill_indices = neighbor_indices == -1
                    edge_features = edge_features[neighbor_idx_hop]
                    node_features = node_features[neighbor_idx_hop]
                    neighbor_masks = neighbor_masks[neighbor_idx_hop]
                    edge_features[fill_indices] = 9999
                    neighbor_indices[fill_indices] = 0
                    neighbor_masks[fill_indices] = 0
                    h = h[:, neighbor_idx_hop, :]
                    c = c[:, neighbor_idx_hop, :]

                graph_feature,node_features, edge_features, (h, c),heads_attention_weights = gnn_layer(
                    node_features, edge_features, neighbor_indices,
                    neighbor_masks, h, c)
            node_features = node_features[mini_batch_hops_indices[-1]]

        '''pooled_features = global_max_pooling(node_features, batch)

        if self.readout_hidden_units is not None:
            pooled_features = self.readout_hidden_layers(pooled_features)

        predictions = self.readout_dropout_layer(self.readout_layer(pooled_features))
        if self.task == "classification":
            predictions = self.logsoftmax(predictions)
        return predictions'''
        # readout
        if self.readout_hidden_units is not None:
            node_features = self.readout_hidden_layers(node_features)

        # 添加图池化操作
        #graph_features = self.pool(node_features)

        predictions = self.readout_layer(self.readout_dropout_layer(graph_feature))
        if self.task == "classification":
            predictions = self.logsoftmax(predictions)
        return predictions,heads_attention_weights[0]
