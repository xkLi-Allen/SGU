import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    parser = argparse.ArgumentParser()
 
    #for all methods#
    parser.add_argument('--cuda', type=int, default=3, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--base_model', type=str, default='SGC', choices=["SIGN", "SGC","S2GC","SAGE", "GAT", 'MLP', "GCN", "GIN","GST"])
    parser.add_argument('--is_transductive', type=bool, default=True, help = "Task is transductive or inductive")
    parser.add_argument('--inductive', type=str, default='normal', choices=['cluster-gcn', 'graphsaint', 'normal'])
    parser.add_argument('--dataset_name', type=str, default='cora',
                        choices=["cora", "citeseer", "pubmed", "CS", "Physics", "flickr", "ppi", "Photo", "Computers","DBLP","ogbl","ogbn-arxiv","ogbn-products"])
    parser.add_argument('--unlearning_methods', type=str, default='GUIM',
                        choices=['GraphEraser', 'GUIDE', 'GNNDelete', 'CEU', "GIF", "GUIM","CGU","GST"])
    parser.add_argument('--exp', type=str, default='attack_unlearning',
                        choices=["partition", "unlearning", "node_edge_unlearning", "attack_unlearning"])
    #train#
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--unlearn_lr', type=float, default=0.02)
    parser.add_argument('--opt_lr', type=float, default=0.001)
    parser.add_argument('--opt_decay', type=float, default=0.0001)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('-l2', type=float, default=1E-5)
    parser.add_argument('-patience', type=int, default=5)
    parser.add_argument('--metrics', type=list, default=["F1_score"], help='Evaluate the models')
    parser.add_argument('--root_path', type=str, default='./', help='Set The Root Path')

    #unlearning parameters#
    parser.add_argument('--num_unlearned_nodes', type=int, default=1)
    parser.add_argument('--proportion_unlearned_nodes', type=float, default=0.1)
    parser.add_argument('--proportion_unlearned_edges', type=str, default="0.0001")
    parser.add_argument('--proportion_unlearned_edges_num', type=float, default=1e-4)
    parser.add_argument('--unlearn_task', type=str, default='node', choices=['feature', "node", "edge"])
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)


    #for shard model#
    parser.add_argument('--num_shards', type=int, default=10)
    parser.add_argument('--test_batch', type=int, default=5)
    parser.add_argument('--train_batch', type=int, default=10)

    #GUIDE parameter
    parser.add_argument('--GUIDE_methods', type=str, default= "Fast",choices=["Fast","SR"])
    parser.add_argument('--GUIDE_repair_methods', type=str, default= "MixUp",choices=["Zero", "Mirror", "MixUp", "NoneR"])


    #GraphEraser parameter
    
    parser.add_argument('--partition_method', type=str, default='lpa',
                        choices=["sage_km", "random", "lpa", "metis", "lpa_base", "sage_km_base"])
    parser.add_argument('--opt_num_epochs', type=int, default=50)
    parser.add_argument('--ratio_deleted_edges', type=float, default=0)
    parser.add_argument('--aggregator', type=str, default='optimal', choices=['mean', 'majority', 'optimal'])
    parser.add_argument('--shard_size_delta', type=float, default=0.005)
    parser.add_argument('--terminate_delta', type=int, default=0)
    parser.add_argument('--is_prune', type=bool, default=False)
    parser.add_argument('--is_partition', type=bool, default=True)
    parser.add_argument('--is_constrained', type=bool, default=True)
    parser.add_argument('--is_train_target_model', type=bool, default=True)
    parser.add_argument('--is_gen_embedding', type=bool, default=True)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--num_opt_samples', type=int, default=100)

    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--use_test_neighbors', type=bool, default=True)
    parser.add_argument('--repartition', type=bool, default=False)

    #GNNDelete parameter
    parser.add_argument('--checkpoint_dir', type=str, default=  './GUIM/data/GNNDelete/checkpoint_node',help='checkpoint folder')
    parser.add_argument('--random_seed', type=int, default=42,help='random seed')
    parser.add_argument('--hidden_dim', type=int, default=64,help='hidden dimension')
    parser.add_argument('--in_dim', type=int, default=128,help='input dimension')
    parser.add_argument('--out_dim', type=int, default=64,help='output dimension')
    parser.add_argument('--unlearning_model', type=str, default='gnndelete_nodeemb',help='unlearning method')
    parser.add_argument('--df', type=str, default='out',help='Df set to use')
    parser.add_argument('--df_idx', type=str, default='none',help='indices of data to be deleted')
    parser.add_argument('--df_size', type=float, default=0.5, help='Df size')
    parser.add_argument('--alpha', type=float, default=0.5,help='alpha in loss function')
    parser.add_argument('--neg_sample_random', type=str, default='non_connected',help='type of negative samples for randomness')
    parser.add_argument('--loss_fct', type=str, default='mse_mean',help='loss function. one of {mse, kld, cosine}')
    parser.add_argument('--loss_type', type=str, default='both_layerwise',help='type of loss. one of {both_all, both_layerwise, only2_layerwise, only2_all, only1}')

    # #CEU
    parser.add_argument('--feature',type=bool,default=False,help='embedding feature')
    parser.add_argument('--feature_update', type=bool, default=True, help='embedding feature update')
    parser.add_argument('--emb_dim', type=int, default=32, help='embedding dim')
    parser.add_argument('-max-degree', action='store_true')
    parser.add_argument('-damping', type=float, default=0.)
    parser.add_argument('-hidden', type=int, nargs='+', default=[])
    parser.add_argument('-approx', type=str, default='cg')
    parser.add_argument('-depth', type=int, default=300)

    # #GIF
    parser.add_argument('--GIF_method', type=str, default="GIF", choices=["GIF", "Retrain", "IF"])
    parser.add_argument('--is_split', type=bool, default=True, help='splitting train/test data')
    parser.add_argument('--iteration', type=int, default=5)
    parser.add_argument('--scale', type=int, default=10000)
    parser.add_argument('--damp', type=float, default=0.0)

    #GUIM
    parser.add_argument('--GNN_layer', type=int, default=3)
    parser.add_argument('--unlearning_epochs', type=int, default=50)
    parser.add_argument('--Budget', type=float, default=0.2)

 
    args = vars(parser.parse_args())
    return args
