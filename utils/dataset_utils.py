import torch
import pickle
import os
import numpy as np
import config
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from config import root_path
from torch_geometric.transforms import SIGN

def process_data(logger,data,args):
    if args["unlearning_methods"] == "CEU":
        data = ceu_process(args,data)
        return data

    if (args['is_transductive']):
        filename = root_path + '/data/processed/transductive/' + args['dataset_name'] + '.pkl'
        if is_data_exists(filename):
            with open(filename, 'rb') as file:
                data = pickle.load(file)
                # data = SIGN(args["GNN_layer"])(data)
                # data.xs = [data.x] + [data[f'x{i}'] for i in range(1, args["GNN_layer"] + 1)]
                # save_data(logger, data, filename)
        else:
            data = transductive_split_node(logger,args,data)
            save_data(logger, data, filename)
    else:
        filename = root_path + '/data/processed/inductive/' + args['dataset_name'] + '.pkl'
        if is_data_exists(filename):
            with open(filename, 'rb') as file:
                data = pickle.load(file)
        else:
            data = inductive_split_node(logger,args,data)
            save_data(logger, data, filename)



    return data

class BasicDataset(Dataset):

    def __init__(self, nodes, labels):
        self.nodes = nodes
        self.labels = labels

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx], self.labels[idx]

    def remove(self, node):
        index = self.nodes.index(node)
        del self.nodes[index]
        del self.labels[index]



def ceu_process(args,data):
    nodes = list(range(len(data.x)))
    edges = [(e[0],e[1]) for e in data.edge_index.t().tolist()]
    features = data.x.numpy()
    num_features = data.x.size(1)
    labels = data.y.tolist()
    num_nodes = data.num_node
    if not args['feature']:
        features = initialize_features(data, num_nodes, args['emb_dim'],args)
        num_features = args['emb_dim']
    train_set_path = os.path.join('./data/CEU', args["dataset_name"], 'train_set.pt')
    valid_set_path = os.path.join('./data/CEU', args["dataset_name"], 'valid_set.pt')
    test_set_path = os.path.join('./data/CEU', args["dataset_name"], 'test_set.pt')
    if os.path.exists(train_set_path) and os.path.exists(valid_set_path) and os.path.exists(test_set_path):
        train_set = torch.load(os.path.join('./data/CEU', args["dataset_name"], 'train_set.pt'))
        valid_set = torch.load(os.path.join('./data/CEU', args["dataset_name"], 'valid_set.pt'))
        test_set = torch.load(os.path.join('./data/CEU', args["dataset_name"], 'test_set.pt'))
    else:
        nodes_train, nodes_test, labels_train, labels_test = train_test_split(nodes, labels, test_size=0.2)
        nodes_train, nodes_valid, labels_train, labels_valid = train_test_split(
            nodes_train, labels_train, test_size=0.2)
        train_set = BasicDataset(nodes_train, labels_train)
        valid_set = BasicDataset(nodes_valid, labels_valid)
        test_set = BasicDataset(nodes_test, labels_test)
        torch.save(train_set, train_set_path)
        torch.save(valid_set, valid_set_path)
        torch.save(test_set, test_set_path)

    data = {
        'nodes': nodes,
        'edges': edges,
        'features': features,
        'labels': labels,
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'num_classes': np.max(labels) + 1,
        'num_features': num_features,
        # 'node2idx': node2idx,
        # 'label2idx': label2idx,
        'train_set': train_set,
        'valid_set': valid_set,
        'test_set': test_set,
    }
    return data


def initialize_features(dataset, num_nodes, emb_dim,args):
    if emb_dim == 32:
        features_path = os.path.join('./data/CEU', args["dataset_name"], 'features.pt')
    else:
        features_path = os.path.join('./data/CEU', args["dataset_name"], f'features_{emb_dim}.pt')
    if os.path.exists(features_path):
        features = torch.load(features_path)
    else:
        features = torch.zeros(num_nodes, emb_dim)
        nn.init.xavier_normal_(features)
        torch.save(features, features_path)
    return features.numpy()



def inductive_split_node(logger,args,data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    create three graph:
    train: data.x[data.train_mask] data.train_edge_index
    val: the same
    test: the same

    :param logger:
    :param args:
    :param data:
    :param train_ratio:
    :param val_ratio:
    :param test_ratio:
    :return:
    """

    num_nodes = data.num_node
    indices = torch.randperm(num_nodes)

    train_mask = indices[:int(train_ratio * num_nodes)]
    val_mask = indices[int(train_ratio * num_nodes):int((train_ratio + val_ratio) * num_nodes)]
    test_mask = indices[int((train_ratio + val_ratio) * num_nodes):]

    # 创建训练、验证和测试集的掩码
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_mask] = 1
    data.val_mask[val_mask] = 1
    data.test_mask[test_mask] = 1

    data.train_indices =data.train_mask.nonzero(as_tuple=True)[0].tolist()
    data.test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    data.val_indices = data.val_mask.nonzero(as_tuple=True)[0].tolist()

    # 复制一份原始的edge_index用于处理连接集合之间的边
    train_edge_index = data.edge_index.clone()
    val_edge_index = data.edge_index.clone()
    test_edge_index = data.edge_index.clone()

    train_new_index = 0
    val_new_index = 0
    test_new_index = 0
    train_dict = {}
    val_dict = {}
    test_dict = {}
    for node in range(num_nodes):
        if data.train_mask[node]:
            train_dict[node] = train_new_index
            train_new_index += 1
        elif data.val_mask[node]:
            val_dict[node] = val_new_index
            val_new_index += 1
        elif data.test_mask[node]:
            test_dict[node] = test_new_index
            test_new_index += 1



    # 删除连接训练、验证和测试集之间的边
    train_edge_mask = data.train_mask[train_edge_index[0]] & data.train_mask[train_edge_index[1]]
    data.train_edge_index = train_edge_index[:, train_edge_mask]
    for edge in range(data.train_edge_index.size(1)):
        data.train_edge_index[0][edge] = train_dict[data.train_edge_index[0][edge].item()]
        data.train_edge_index[1][edge] = train_dict[data.train_edge_index[1][edge].item()]
    # 删除连接验证集之外的边
    val_edge_mask = data.val_mask[val_edge_index[0]] & data.val_mask[val_edge_index[1]]
    data.val_edge_index = val_edge_index[:, val_edge_mask]
    for edge in range(data.val_edge_index.size(1)):
        data.val_edge_index[0][edge] = val_dict[data.val_edge_index[0][edge].item()]
        data.val_edge_index[1][edge] = val_dict[data.val_edge_index[1][edge].item()]

        # 删除连接测试集之外的边
    test_edge_mask = data.test_mask[test_edge_index[0]] & data.test_mask[test_edge_index[1]]
    data.test_edge_index = test_edge_index[:, test_edge_mask]
    for edge in range(data.test_edge_index.size(1)):
        data.test_edge_index[0][edge] = test_dict[data.test_edge_index[0][edge].item()]
        data.test_edge_index[1][edge] = test_dict[data.test_edge_index[1][edge].item()]


    save_train_test_split(logger,args,data.train_indices,data.test_indices)

    return data


def transductive_split_node(logger,args,data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    num_nodes = data.num_node
    indices = torch.randperm(num_nodes)

    train_mask = indices[:int(train_ratio * num_nodes)]
    val_mask = indices[int(train_ratio * num_nodes):int((train_ratio + val_ratio) * num_nodes)]
    test_mask = indices[int((train_ratio + val_ratio) * num_nodes):]

    # 创建训练、验证和测试集的掩码
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_mask] = 1
    data.val_mask[val_mask] = 1
    data.test_mask[test_mask] = 1

    data.train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    data.test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    data.val_indices = data.val_mask.nonzero(as_tuple=True)[0].tolist()

    data.train_edge_index = data.edge_index.clone()
    data.val_edge_index = data.edge_index.clone()
    data.test_edge_index = data.edge_index.clone()
    # save_train_test_split(logger,args, data.train_indices, data.test_indices)

    return data


def c2n_to_n2c(args, community_to_node):
    node_list = []
    for i in range(args['num_shards']):
        node_list.extend(list(community_to_node.values())[i])
    node_to_community = {}

    for comm, nodes in dict(community_to_node).items():
        for node in nodes:
            # Map node id back to original graph
            # node_to_community[node_list[node]] = comm
            node_to_community[node] = comm

    return node_to_community





def save_train_test_split(logger,args,train_indices, test_indices):
    train_test_split_file = config.train_test_split_file
    os.makedirs(os.path.dirname(train_test_split_file), exist_ok=True)
    logger.info("save_train_test_split:{}".format(train_test_split_file) )
    pickle.dump((train_indices, test_indices), open(train_test_split_file, 'wb'))

def save_data(logger,data, filename):
    logger.info("save_data {}".format(filename))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def save_train_data(logger,data,filename):
    logger.info("save_train_data {}".format(filename))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def save_train_graph(logger,gragh_data,filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(gragh_data, file)

def save_unlearned_data(logger,data, suffix):
    logger.info('saving unlearned data {}'.format('_'.join((config.unlearned_file, suffix))))
    pickle.dump(data, open('_'.join((config.unlearned_file, suffix)), 'wb'))


def load_unlearned_data(logger, suffix):
    file_path = '_'.join((config.unlearned_file, suffix))
    logger.info('loading unlearned data from %s' % file_path)
    return pickle.load(open(file_path, 'rb'))

def save_community_data(logger,community_to_node,filename, suffix=''):
    logger.info('save_community_data {}'.format(filename + suffix))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump(community_to_node, open(filename + suffix, 'wb'))

def save_shard_data(logger,shard_data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump(shard_data, open(filename, 'wb'))

def load_shard_data(logger):
    logger.info("load_shard_data {}".format(config.shard_file))
    return pickle.load(open(config.shard_file, 'rb'))


def load_train_graph(logger):
    logger.info("load_train_graph {}".format(config.train_graph_file))
    return pickle.load(open(config.train_graph_file, 'rb'))

def save_embeddings(embeddings,filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump(embeddings, open(filename, 'wb'))

def load_community_data(logger,filename = config.load_community_data,suffix=''):
    logger.info("load_community_data {}".format(filename+suffix))
    return pickle.load(open(filename + suffix, 'rb'))

def load_saved_data(logger,filename = config.train_data_file):
    logger.info('load_saved_data {}'.format(filename))
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def load_embeddings(filename):
    with open(filename, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

def load_train_test_split(logger, filename = config.train_test_split_file):
    logger.info("load_train_test_split {}".format(filename))
    return  pickle.load(open(filename, 'rb'))

def is_data_exists(filename):
    return os.path.exists(filename)

def _extract_embedding_method(partition_method):
    return partition_method.split('_')[0]

def save_target_model(logger,args,run, model, shard, suffix=''):

    if args["exp"] in ["node_edge_unlearning", "attack_unlearning"]:
        model_path = '_'.join((config.target_model_file, str(shard), str(run), str(args['num_unlearned_nodes']))) + suffix
        model.save_model(model_path)
        logger.info("save_target_model {}".format(model_path))
    else:
        model.save_model(config.target_model_file + '_' + str(shard) + '_' + str(run))
        logger.info("save_target_model {}".format(config.target_model_file + '_' + str(shard) + '_' + str(run)))
        # model.save_model(self.target_model_file + '_' + str(shard))

def load_target_model(logger,args, run, model, shard, suffix=''):
    if args["exp"] == "node_edge_unlearning":
        model.load_model(
            '_'.join((config.target_model_file, str(shard), str(run), str(args['num_unlearned_nodes']))))
        # logger.info("load_target_model {}".format('_'.join((config.target_model_file, str(shard), str(run), str(args['num_unlearned_nodes'])))))
    elif args["exp"] == "attack_unlearning":
        if(suffix == ''):
            model_path = '_'.join((config.target_model_file, str(shard), str(run)))
        else:
            model_path = '_'.join((config.target_model_file, str(shard), str(run), str(args['num_unlearned_nodes']))) + suffix
        device = torch.device('cpu')
        model.load_model(model_path)
        model.device=device
        # logger.info("load_target_model {}".format(model_path))
    else:
        # model.load_model(self.target_model_file + '_' + str(shard) + '_' + str(run))
        model.load_model(config.target_model_file + '_' + str(shard) + '_' + str(0))
        # logger.info("load_target_model {}".format(config.target_model_file + '_' + str(shard) + '_' + str(0)))

def save_posteriors(logger,args, posteriors, run, suffix=''):
    posteriors_path = config.ANALYSIS_PATH + 'posteriors/' + args['dataset_name'] + '/' + config.target_model_name
    os.makedirs(os.path.dirname(posteriors_path), exist_ok=True)
    logger.info("save_posteriors {}".format(posteriors_path))
    torch.save(posteriors, posteriors_path + '_' + str(run) + suffix)

def save_optimal_weight(logger,args,weight,run):
    optimal_weight_path = config.ANALYSIS_PATH + 'optimal/' + args['dataset_name'] + '/' + config.optimal_weight_name
    os.makedirs(os.path.dirname(optimal_weight_path), exist_ok=True)
    torch.save(weight, optimal_weight_path + '_' + str(run))