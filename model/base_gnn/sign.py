
from torch.utils.data import DataLoader
from torch_geometric.transforms import SIGN
import torch
import torch.nn.functional as F
from model.base_gnn.abstract_model import abstract_model
from config import root_path
from typing import Union
from torch_geometric.typing import Adj, OptTensor, OptPairTensor
from torch import Tensor
from parameter_parser import parameter_parser
args = parameter_parser()

class SIGNNet(abstract_model):
    def __init__(self, in_channels, out_channels, num_layers = 3,
                 dropout=0):
        super(SIGNNet, self).__init__()

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers + 1):
            self.lins.append(torch.nn.Linear(in_channels, 64,bias=False))
        self.lin1 = torch.nn.Linear((num_layers + 1) * 64,64, bias=False)

        #########new#########
        self.lin2 = torch.nn.Linear(64,out_channels, bias=False)


        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, xs):
        xs = xs.transpose(0, 1)
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.lin1(x)
        x = self.lin2(x)
        if args["unlearning_methods"] == "GIF":
            return F.log_softmax(x, dim=1)
        else:
            return x

    def forward_GUIM(self,xs):
        xs = xs.transpose(0, 1)
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)

        return x

    def get_features(self, xs):
        xs = xs.transpose(0, 1)
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        return x


    def get_softlabel(self,xs):
        xs = xs.transpose(0, 1)
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.lin1(x)
        x = self.lin2(x)

        return F.softmax(x,dim=1)
    def emb2softlable(self,x):
        x = self.lin2(x)

        return F.softmax(x,dim=1)


    def get_embedding(self,xs):
        xs = xs.transpose(0, 1)
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        emb = self.lin1(x)
        return emb
    
    def forward_once(self, data):
        data.edge_index = data.edge_index.contiguous()
        data = SIGN(args["GNN_layer"])(data)
        data.xs = [data.x] + [data[f'x{i}'] for i in range(1, args["GNN_layer"] + 1)]
        data.xs = torch.tensor([x.detach().cpu().numpy() for x in data.xs]).cuda()
        xs = data.xs
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.lin1(x)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

    def forward_once_unlearn(self, data):
        xs = data.xs_unlearn.transpose(0,1)
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.lin1(x)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)
    
    def GIF_inference(self, data):
        xs = data.xs
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x

    