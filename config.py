from parameter_parser import parameter_parser
args = parameter_parser()

##for Graph Eraser###
RAW_DATA_PATH = './bwfan/GUIM/data/raw/'
PROCESSED_DATA_PATH = './bwfan/GUIM/data/GraphEraser/processed/'
if args["unlearning_methods"] == "GIF":
    PROCESSED_DATA_PATH2 = './bwfan/GUIM/data/processed/GIF/'
else:
    PROCESSED_DATA_PATH2 = './bwfan/GUIM/data/processed/transductive/'
MODEL_PATH = './bwfan/GUIM/data/GraphEraser/'
ANALYSIS_PATH = './bwfan/GUIM/data/GraphEraser/analysis_data/'
# RAW_DATA_PATH = './data/raw/'
# PROCESSED_DATA_PATH = './data/GraphEraser/processed/'
# PROCESSED_DATA_PATH2 = './data/processed/transductive/'
# MODEL_PATH = './data/GraphEraser/'
# ANALYSIS_PATH = './data/GraphEraser/analysis_data/'
community_name = '_'.join(('community', args['partition_method'], str( args['num_shards']),
                           str(args['ratio_deleted_edges'])))
shard_name = '_'.join(('shard_data', args['partition_method'], str(args['num_shards']),
                       str(args['shard_size_delta']), str(args['ratio_deleted_edges'])))
target_model_name = '_'.join((args['base_model'], args['partition_method'], str( args['num_shards']),
                              str(args['shard_size_delta']), str(args['ratio_deleted_edges'])))
optimal_weight_name = '_'.join((args['base_model'],args['partition_method'], str(args['num_shards']),
                                str(args['shard_size_delta']), str(args['ratio_deleted_edges'])))
processed_data_prefix = PROCESSED_DATA_PATH + args['dataset_name'] + "/"
shard_file = processed_data_prefix + shard_name
train_data_file = processed_data_prefix + "train_data"
train_graph_file = processed_data_prefix + "train_graph"
# train_test_split_file = processed_data_prefix + "/split/"+"train_test_split_" + str(
#         args['test_ratio'])
train_test_split_file = PROCESSED_DATA_PATH2 + args['dataset_name'] + ".pkl"
load_community_data = processed_data_prefix +community_name
community_file = processed_data_prefix + community_name
community_path = PROCESSED_DATA_PATH + args['dataset_name'] + "/" + community_name
unlearned_file = processed_data_prefix+ '_'.join(('unlearned', str(args['num_unlearned_nodes'])))
model_path =MODEL_PATH + args['dataset_name'] + "/" + target_model_name
GIF_logger_name = "_".join((args['dataset_name'], str(args['test_ratio']), args['base_model'], args['unlearn_task'],
                        str(args['unlearn_ratio'])))

target_model_file = MODEL_PATH + args['dataset_name'] + '/' + target_model_name


#for GUIM
root_path = "./bwfan/GUIM"
# root_path = "."
unlearning_path = root_path + "/data/unlearning_nodes_" + str(args["proportion_unlearned_nodes"]) + "_" + args["dataset_name"] + ".txt"

