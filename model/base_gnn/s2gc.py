import torch
import yaml
from torch_geometric.nn import SGConv
from typing import Union
import torch.nn.functional as F

from model.base_gnn.Convs import S2GConv
from model.base_gnn.abstract_model import abstract_model
from torch_geometric.typing import Adj, OptTensor, OptPairTensor
from torch import Tensor
from config import root_path
from parameter_parser import parameter_parser
args = parameter_parser()

class S2GCNet(abstract_model):
    def __init__(self, in_channels, out_channels, num_layers=3):
        super(S2GCNet, self).__init__()
        self.config = self.load_config()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        if args["unlearning_methods"] == "GUIM" or args["unlearning_methods"] =="GNNDelete":
            # self.convs.append(SGConv(in_channels, 64, K=3, bias=False))
            self.convs.append(torch.nn.Linear(in_channels, 64, bias=False))
            self.convs.append(torch.nn.Linear(64,out_channels,bias=False))
        else:
            if args["unlearning_methods"] != "GIF":
                self.convs.append(S2GConv(in_channels, 64, K=self.num_layers, bias=False))
                self.convs.append(torch.nn.Linear(64, out_channels, bias=False))
            else:
                self.num_layers = 2
                self.convs_batch = torch.nn.ModuleList()
                self.convs_batch.append(S2GConvBatch(in_channels, 16, cached=False, add_self_loops=True, bias=False))
                self.convs_batch.append(S2GConvBatch(16, out_channels, cached=False, add_self_loops=True, bias=False))

    def forward(self, x, edge_index = None,adjs=None,edge_weight=None):
        if args["unlearning_methods"] == "GIF":
            for i, (edge_index_, e_id, size) in enumerate(adjs):
                    x_target = x[:size[1]]  # Target nodes are always placed first.
                    x = self.convs_batch[i]((x, x_target), edge_index_, edge_weight=edge_weight[e_id])

                    if i != self.num_layers - 1:
                        x = F.dropout(x, p=0.5, training=self.training)

            return F.log_softmax(x, dim=1)
        else:
            if edge_index is None:
                x = self.convs[0](x)
                x = self.convs[1](x)
            else:
                x = self.convs[0](x,edge_index)

            return x

    def get_softlabel(self,x):
        x = self.convs[0](x)
        x = self.convs[1](x)

        return F.softmax(x,dim=1)
    def emb2softlable(self,x):
        x = self.convs[1](x)

        return F.softmax(x,dim=1)


    def get_embedding(self,x):
        emb = self.convs[0](x)
        return emb

    def reset_parameters(self):
        if args["unlearning_methods"] == "GUIM":
            for i in range(2):
                self.convs[i].reset_parameters()
        else:
            if args["unlearning_methods"] != "GIF":
                self.convs[0].reset_parameters()
            else:
                self.convs_batch[0].reset_parameters()
                self.convs_batch[1].reset_parameters()

    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all



    def GIF_inference(self, x_all, subgraph_loader, edge_weight, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs_batch[i]((x, x_target), edge_index, edge_weight=edge_weight[e_id])

                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
    
    def forward_once(self, data, edge_weight):
        x, edge_index = data.x, data.edge_index
        x = self.convs_batch[0](x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs_batch[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def forward_once_unlearn(self, data, edge_weight):
        x, edge_index = data.x_unlearn, data.edge_index_unlearn
        x = self.convs_batch[0](x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs_batch[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)
    
    def load_config(self):
        f = open(root_path + '/model/properties/s2gc' +'.yaml','r')
        config_str = f.read()
        config = yaml.load(config_str,Loader=yaml.FullLoader)
        self.lr = config['lr']
        self.decay = config['decay']
        return config
    
class S2GConvBatch(S2GConv):
    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True, 
                 bias: bool = True, **kwargs):
        super(S2GConvBatch, self).__init__(in_channels, out_channels,
                                          cached=cached, add_self_loops=add_self_loops,
                                          bias=bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        out = self.lin(out)

        return out