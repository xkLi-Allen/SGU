import copy
import math
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import networkx as nx
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score,recall_score
from model.base_gnn.ceu_model import CEU_GNN
from tqdm import tqdm
from scipy.stats import entropy
from sklearn import preprocessing
from numpy.linalg import norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch import Tensor
from typing import Optional
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.metrics import roc_curve, auc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def connected_component_subgraphs(graph):
    """
    Find all connected subgraphs in a networkx Graph

    Args:
        graph (Graph): A networkx Graph

    Yields:
        generator: A subgraph generator
    """
    for c in nx.connected_components(graph):
        yield graph.subgraph(c)

def filter_edge_index_1(data, node_indices):
    """
    Remove unnecessary edges from a torch geometric Data, only keep the edges between node_indices.
    Args:
        data (Data): A torch geometric Data.
        node_indices (list): A list of nodes to be deleted from data.

    Returns:
        data.edge_index: The new edge_index after removing the node_indices.
    """
    if isinstance(data.edge_index, torch.Tensor):
        data.edge_index = data.edge_index.cpu()

    edge_index = data.edge_index
    node_index = np.isin(edge_index, node_indices)

    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = data.edge_index[:, col_index]

    return np.searchsorted(node_indices, edge_index)

def filter_edge_index(edge_index, node_indices, reindex=True):
    assert np.all(np.diff(node_indices) >= 0), 'node_indices must be sorted'
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu()

    node_index = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = edge_index[:, col_index]

    if reindex:
        return np.searchsorted(node_indices, edge_index)
    else:
        return edge_index


@torch.no_grad()
def negative_sampling_kg(edge_index, edge_type):
    '''Generate negative samples but keep the node type the same'''

    edge_index_copy = edge_index.clone()
    for et in edge_type.unique():
        mask = (edge_type == et)
        old_source = edge_index_copy[0, mask]
        new_index = torch.randperm(old_source.shape[0])
        new_source = old_source[new_index]
        edge_index_copy[0, mask] = new_source

    return edge_index_copy

#for GNNDelete
def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    return torch.cat([row[mask], col[mask]], dim=0)


def get_loss_fct(name):
    # if name == 'mse':
    #     loss_fct = nn.MSELoss(reduction='mean')
    # elif name == 'kld':
    #     loss_fct = BoundedKLDMean
    # elif name == 'cosine':
    #     loss_fct = CosineDistanceMean

    if name == 'kld_mean':
        loss_fct = BoundedKLDMean
    elif name == 'kld_sum':
        loss_fct = BoundedKLDSum
    elif name == 'mse_mean':
        loss_fct = nn.MSELoss(reduction='mean')
    elif name == 'mse_sum':
        loss_fct = nn.MSELoss(reduction='sum')
    elif name == 'cosine_mean':
        loss_fct = CosineDistanceMean
    elif name == 'cosine_sum':
        loss_fct = CosineDistanceSum
    elif name == 'linear_cka':
        loss_fct = LinearCKA
    elif name == 'rbf_cka':
        loss_fct = RBFCKA
    else:
        raise NotImplementedError

    return loss_fct

def BoundedKLDMean(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'batchmean'))

def BoundedKLDSum(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'sum'))

def CosineDistanceMean(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).mean()

def CosineDistanceSum(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).sum()

def LinearCKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def RBFCKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))
    return hsic / (var1 * var2)


def kernel_HSIC(X, Y, sigma=None):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)

def rbf(X, sigma=None):
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def trange(*args, **kwargs):
    """Shortcut for tqdm(range(*args), **kwargs)."""
    return tqdm(range(*args), **kwargs)


#CEU
def remove_undirected_edges(edges, edges_to_remove):
    _edges = set(copy.deepcopy(edges))
    for e in edges_to_remove:
        if not isinstance(e, tuple):
            e = tuple(e)
        if e in _edges:
            _edges.remove(e)
        if (e[1], e[0]) in _edges:
            _edges.remove((e[1], e[0]))
    return list(_edges)

def CEU_load_model(args, data, type='original', edges=None, edge=None, node=None):
    assert type in ['original', 'edge', 'node', 'retrain', 'unlearn'], f'Invalid type of model, {type}'
    if type == 'edge':
        model = CEU_create_model(args, data)
        model.load_state_dict(torch.load(os.path.join('./checkpoint', args.data, 'edges',
                              f'{args.model}_{args.data}_{edge[0]}_{edge[1]}_best.pt')))
        return model
    elif type == 'node':
        model = CEU_create_model(args, data)
        model.load_state_dict(torch.load(os.path.join('./checkpoint', args.data, 'nodes',
                              f'{args.model}_{args.data}_{node}_best.pt')))
        return model
    else:
        model = CEU_create_model(args, data)
        model.load_state_dict(torch.load(CEU_model_path(args, type, edges)))
        return model

def CEU_create_model(args, data):
    embedding_size = args.emb_dim if data['features'] is None else data['features'].shape[1]
    model = CEU_GNN(data['num_nodes'], embedding_size,
                args.hidden, data['num_classes'], data['features'], args.feature_update, args.model)
    return model

def CEU_model_path(args, type, edges=None):
    if args["hidden"]:
        layers = '-'.join([str(h) for h in args.hidden])
        prefix = f'{args.model}_{args.data}_{layers}'
    else:
        prefix = f'{args.model}_{args.data}'

    if type == 'original':
        return os.path.join('./checkpoint/CEU', args.data, f'{prefix}_best.pt')
    elif type == 'retrain':
        if args["max_degree"]:
            filename = f'{prefix}_{type}_max_{args.method}{edges}_best.pt'
        else:
            filename = f'{prefix}_{type}_{args.method}{edges}_best.pt'
        return os.path.join('./checkpoint/CEU', args.data, filename)
    elif type == 'unlearn':
        assert edges is not None
        if args.batch_unlearn:
            prefix += '_batch'
        if args.unlearn_batch_size is not None:
            prefix += f'args.unlearn_batch_size'
        if args.approx == 'lissa':
            filename = f'{prefix}_{type}_{args["unlearning_methods"]}{edges}_{args["approx"]}d{args["depth"]}r{args.r}_best.pt'
        else:
            filename = f'{prefix}_{type}_{args["unlearning_methods"]}{edges}_{args["approx"]}_d{args["damping"]}_best.pt'
        return os.path.join('./checkpoint/CEU', args["dataset_name"], filename)
    else:
        raise ValueError('Invalid type of model,', type)

def JSD(P, Q):
    _M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, _M, axis=1) + entropy(Q, _M, axis=1))

##################for CGU###############

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, test_lb=1000, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    if Flag == 0:
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    else:
        all_index = torch.randperm(data.y.shape[0])
        data.val_mask = index_to_mask(all_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(all_index[val_lb: (val_lb+test_lb)], size=data.num_nodes)
        data.train_mask = index_to_mask(all_index[(val_lb+test_lb):], size=data.num_nodes)
    return data

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def preprocess_data(X):
    '''
    input:
        X: (n,d), torch.Tensor
    '''
    X_np = X.cpu().numpy()
    scaler = preprocessing.StandardScaler().fit(X_np)
    X_scaled = scaler.transform(X_np)
    row_norm = norm(X_scaled, axis=1)
    X_scaled = X_scaled / row_norm.max()
    return torch.from_numpy(X_scaled)


class MyGraphConv(MessagePassing):
    """
    Use customized propagation matrix. Just PX (or PD^{-1}X), no linear layer yet.
    """
    _cached_x: Optional[Tensor]

    def __init__(self, K: int = 1,
                 add_self_loops: bool = True,
                 alpha=0.5, XdegNorm=False, GPR=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.K = K
        self.add_self_loops = add_self_loops
        self.alpha = alpha
        self.XdegNorm = XdegNorm
        self.GPR = GPR
        self._cached_x = None  # Not used
        self.reset_parameters()

    def reset_parameters(self):
        self._cached_x = None  # Not used

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)
        elif isinstance(edge_index, SparseTensor):
            edge_index = get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)

        if self.XdegNorm:
            # X <-- D^{-1}X, our degree normalization trick
            num_nodes = maybe_num_nodes(edge_index, None)
            row, col = edge_index[0], edge_index[1]
            deg = degree(row).unsqueeze(-1)

            deg_inv = deg.pow(-1)
            deg_inv = deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        if self.GPR:
            xs = []
            xs.append(x)
            if self.XdegNorm:
                x = deg_inv * x  # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                xs.append(x)
            return torch.cat(xs, dim=1) / (self.K + 1)
        else:
            if self.XdegNorm:
                x = deg_inv * x  # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')

# prepare P matrix in PyG format
def get_propagation(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None,
                    alpha=0.5):
    """
    return:
        P = D^{-\alpha}AD^{-(1-alpha)}.
    """
    fill_value = 2. if improved else 1.
    assert (0 <= alpha) and (alpha <= 1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_left = deg.pow(-alpha)
    deg_inv_right = deg.pow(alpha - 1)
    deg_inv_left.masked_fill_(deg_inv_left == float('inf'), 0)
    deg_inv_right.masked_fill_(deg_inv_right == float('inf'), 0)

    return edge_index, deg_inv_left[row] * edge_weight * deg_inv_right[col]


# training iteration for binary classification
def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS', lr=0.01, wd=0,
                X_val=None, y_val=None):
    '''
        b is the noise here. It is either pre-computed for worst-case, or pre-defined.
    '''
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)

    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)

    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise ("Error: Not supported optimizer.")

    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()

        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i + 1, loss.cpu(), w.grad.norm()))

        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise ("Error: Not supported optimizer.")

        # If we want to control the norm of w_best, we should keep the last w instead of the one with
        # the highest val acc
        if X_val is not None:
            val_acc = lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best


# loss for binary classification
def lr_loss(w, X, y, lam):
    '''
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        averaged training loss with L2 regularization
    '''
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2


# evaluate function for binary classification
def lr_eval(w, X, y):
    '''
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
    return:
        prediction accuracy
    '''
    return X.mv(w).sign().eq(y).float().mean()


# gradient of loss wrt w for binary classification
def lr_grad(w, X, y, lam):
    '''
    The gradient here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        gradient: (d,)
    '''
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z - 1) * y) + lam * X.size(0) * w


# hessian of loss wrt w for binary classification
def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    '''
    The hessian here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
        batch_size: int
    return:
        hessian: (d,d)
    '''
    z = torch.sigmoid(y * X.mv(w))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()


# training iteration for binary classification
def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS', lr=0.01, wd=0,
                X_val=None, y_val=None):
    '''
        b is the noise here. It is either pre-computed for worst-case, or pre-defined.
    '''
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)

    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)

    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise ("Error: Not supported optimizer.")

    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()

        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i + 1, loss.cpu(), w.grad.norm()))

        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise ("Error: Not supported optimizer.")

        # If we want to control the norm of w_best, we should keep the last w instead of the one with
        # the highest val acc
        if X_val is not None:
            val_acc = lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best


# aggregated loss for multiclass classification
def ovr_lr_loss(w, X, y, lam, weight=None):
    '''
     input:
        w: (d,c)
        X: (n,d)
        y: (n,c), one-hot
        lambda: scalar
        weight: (c,) / None
    return:
        loss: scalar
    '''
    z = batch_multiply(X, w) * y
    if weight is None:
        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
    else:
        return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2


def ovr_lr_eval(w, X, y):
    '''
    input:
        w: (d,c)
        X: (n,d)
        y: (n,), NOT one-hot
    return:
        loss: scalar
    '''
    pred = X.mm(w).max(1)[1]
    # softlabel = F.softmax(X.mm(w))
    # y_true = torch.zeros(y.size(0),7).cpu()
    # y_index = y.view(y.size(0),-1).cpu()
    # y_true = y_true.scatter_(1, y_index, 1)
    F1_score = f1_score( y.cpu(), pred.cpu(), average="micro")
    Recall_score = recall_score(y.cpu(), pred.cpu(),  average="micro")

    return pred.eq(y).float().mean(),F1_score,Recall_score


def ovr_lr_optimize(X, y, lam, weight=None, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS',
                    lr=0.01, wd=0, X_val=None, y_val=None):
    '''
    y: (n_train, c). one-hot
    y_val: (n_val,) NOT one-hot
    '''
    # We use random initialization as in common DL literature.
    # w = torch.zeros(X.size(1), y.size(1)).float()
    # init.kaiming_uniform_(w, a=math.sqrt(5))
    # w = torch.autograd.Variable(w.to(device), requires_grad=True)
    # zero initialization
    w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(device), requires_grad=True)

    def closure():
        if b is None:
            return ovr_lr_loss(w, X, y, lam, weight)
        else:
            return ovr_lr_loss(w, X, y, lam, weight) + (b * w).sum() / X.size(0)

    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise ("Error: Not supported optimizer.")

    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = ovr_lr_loss(w, X, y, lam, weight)
        if b is not None:
            if weight is None:
                loss += (b * w).sum() / X.size(0)
            else:
                loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
        loss.backward()

        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i + 1, loss.cpu(), w.grad.norm()))

        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise ("Error: Not supported optimizer.")

        if X_val is not None:
            val_acc = ovr_lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise ("Error: Training procedure failed")
    return w_best

def batch_multiply(A, B, batch_size=500000):
    if A.is_cuda:
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i+1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub.to(device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        return torch.cat(out, dim=0).to(device)


def get_worst_Gbound_feature(lam, m, deg_m, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return gamma2 * ((2*c*lam + (c*gamma1+lam*c1)*deg_m) ** 2) / (lam ** 4) / (m-1)


def get_worst_Gbound_edge(lam, m, K, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return 16 * gamma2 * (K**2) * ((c*gamma1+lam*c1) ** 2) / (lam ** 4) / m


def get_worst_Gbound_node(lam, m, K, deg_m, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return gamma2 * ((2*c*lam + K*(c*gamma1+lam*c1)*(2*deg_m-1)) ** 2) / (lam ** 4) / (m-1)


def get_c(delta):
    return np.sqrt(2*np.log(1.5/delta))


def get_budget(std, eps, c):
    return std * eps / c

# K = X^T * X for fast computation of spectral norm
def get_K_matrix(X):
    K = X.t().mm(X)
    return K

# using power iteration to find the maximum eigenvalue
def sqrt_spectral_norm(A, num_iters=100):
    '''
    return:
        sqrt of maximum eigenvalue/spectral norm
    '''
    x = torch.randn(A.size(0)).float().to(device)
    for i in range(num_iters):
        x = A.mv(x)
        x_norm = x.norm()
        x /= x_norm
    max_lam = torch.dot(x, A.mv(x)) / torch.dot(x, x)
    return math.sqrt(max_lam)


################GUIM###################
import scipy.sparse as sp
def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()



def cal_distance(vector1,vector2):
    # vector1 = vector1.reshape(1,-1)
    # vector2 = vector2.reshape(1, -1)
    #cos similarity
    # similarity = F.cosine_similarity(vector1,vector2,dim=0)
    similarity = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similarity


def Reverse_CE(out,y):

    return 0


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def plot_auc(y_true,y_score):
    y_true = y_true
    y_score = y_score

    # 计算ROC曲线上的点
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # 计算AUC
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def remove_node_from_graph(data, node_id=None, removal_queue=None):
    """
    Node removal for graph classification. datalist is a list of Data objects. Each Data object corresponds to one training graph.
    
    If graph/node id are provided, remove accordingly.
    
    Otherwise remove a random node from a random graph.
    
    Can optionally record the removal queue.
    """

    # if graph_id is None:
    #     graph_id = np.random.choice(len(datalist))
    #     while datalist[graph_id].empty:
    #         # Ensure we do not further remove from an empty graph!
    #         graph_id = np.random.choice(len(datalist))
    
    # data = datalist[graph_id]
    
    # if node_id is None:
    #     # data.mask records the remaining valid nodes in the graph
    #     node_id = np.random.choice(torch.arange(data.x.shape[0])[data.mask])

    # Editing graph
    data.mask = torch.ones((data.x.shape[0],)).bool()
    data.empty = False
    data.x[node_id] = torch.zeros_like(data.x[node_id])
    data.mask[node_id] = False
    edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
    edge_mask[data.edge_index[0] == node_id] = False
    edge_mask[data.edge_index[1] == node_id] = False
    data.edge_index = data.edge_index[:, edge_mask]
    if data.edge_weight is not None:
        data.edge_weight = data.edge_weight[:, edge_mask]
    
    # Put edited graph back
    # if torch.any(data.mask):
    #     # Still have nodes for this graph, just put it back
    #     datalist[graph_id] = data
    # else:
    #     # Becomes an empty graph
    #     datalist[graph_id].empty = True
    #     # This ensure the masked feature matrix does not becomes empty! Still, the feature is already removed (i.e. zeros)
    #     data.mask[0] = True
    #     # Now, put the graph back.
    #     datalist[graph_id] = data
    
    # if removal_queue is not None:
    #     removal_queue.append([graph_id, node_id])
    #     return datalist, removal_queue
    
    return data