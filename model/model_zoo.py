import torch
import os
import numpy as np
import yaml
import torch.nn.functional as F
import torch_geometric.transforms as T
from utils import utils
from model.base_gnn.graphsage import SAGENet
from model.base_gnn.gat import GATNet
from model.base_gnn.gcn import GCNNet
from model.base_gnn.gin import GINNet
from model.base_gnn.sgc import SGCNet
from model.base_gnn.s2gc import S2GCNet
from model.base_gnn.sign import SIGNNet
from .base_gnn.deletion import GCNDelete, GATDelete, GINDelete,SAGEDelete,SGCDelete,S2GCDelete,SIGNDelete
from model.base_gnn.ceu_model import CEU_GNN
from torch_geometric.data import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from model.base_gnn.abstract_model import abstract_model
from config import root_path


class model_zoo(abstract_model):
    def __init__(self,args,data):
        super(model_zoo, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.data = data
        self.determine_model()
        if self.args["unlearning_methods"] != "GST":
            self.model_config = model_config(args)


    def determine_model(self):
        if self.args['unlearning_methods']=="CEU":
            self.model = self.CEU_get_model().to(self.device)
        else:
            self.model = self.init_model(self.data.num_features, self.data.num_classes).to(self.device)


    def init_model(self, num_feats, num_classes):
        if self.args['base_model'] == 'SAGE':
            return SAGENet(num_feats, num_classes)
        elif self.args['base_model'] == 'GAT':
            return GATNet(num_feats, num_classes)
        elif self.args['base_model'] == 'GCN':
            return GCNNet(num_feats, num_classes)
        elif self.args['base_model'] == 'GIN':
            return GINNet(num_feats, num_classes)
        elif self.args['base_model'] == 'SGC':
            return SGCNet( num_feats, num_classes,num_layers=3)
        elif self.args['base_model'] == 'S2GC':
            return S2GCNet( num_feats, num_classes,num_layers=3)
        elif self.args['base_model'] == 'SIGN':
            return SIGNNet(num_feats, num_classes, num_layers=3)
        else:
            raise Exception('unsupported target models')

    #def load_model(self, save_path):
        # self.logger.info('loading models')
        #device = torch.device('cpu')
        #self.model.load_state_dict(torch.load(save_path, map_location=device))


    #Code for GNNDelete
    def get_model(self,mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None):
        #print("get_model:{}".format(self.args["unlearning_model"]))
        if 'gnndelete' in self.args["unlearning_model"]:
            model_mapping = {'GCN': GCNDelete, 'GAT': GATDelete, 'GIN': GINDelete,'SAGE':SAGEDelete,"SGC":SGCDelete,"S2GC":S2GCDelete,"SIGN":SIGNDelete}
            return model_mapping[self.args["base_model"]](self.args,self.data.num_features, self.data.num_classes,
                                                   mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes)
        else:
            model_mapping = {'GCN': GCNNet, 'GAT': GATNet, 'GIN': GINNet,'SAGE':SAGENet,"SGC":SGCNet,"S2GC":S2GCNet,"SIGN":SIGNNet}
            return model_mapping[self.args["base_model"]](self.data.num_features, self.data.num_classes)

    # Code for GNNDelete
    def CEU_get_model(self):
        embedding_size = self.args['emb_dim'] if self.data['features'] is None else self.data['features'].shape[1]
        model = CEU_GNN(self.data['num_nodes'], embedding_size,
                        self.args["hidden"], self.data['num_classes'], self.data['features'],
                        self.args["feature_update"], self.args["base_model"])
        return model


class model_config:
    def __init__(self,args):
        self.load_config(args)



    def load_config(self,args):
        f = open(root_path + '/model/properties/' + args['base_model'].lower()+'.yaml','r')
        config_str = f.read()
        config = yaml.load(config_str,Loader=yaml.FullLoader)
        self.lr = config['lr']
        self.decay = config['decay']
        return config