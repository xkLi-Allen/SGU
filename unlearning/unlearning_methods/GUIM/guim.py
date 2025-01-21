import random
import heapq
import numpy
import torch
import numpy as np
from numba import njit, prange, jit
import os
import torch.nn.functional as F
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score,recall_score,roc_auc_score
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_undirected
import time
from task.node_classification import NodeClassifier
from utils import utils
from utils.utils import aug_normalized_adjacency, cal_distance
from sklearn.utils import shuffle
from torch_geometric.transforms import SIGN
from config import root_path,unlearning_path
from model.base_gnn.Convs import S2GConv,SGConv,GCNConv


class guim():
    def __init__(self,args,logger,model_zoo):
        self.args = args
        self.logger = logger
        self.model_zoo = model_zoo
        self.data = model_zoo.data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = 0.8
        self.Budget = int(self.data.num_nodes *self.args["Budget"])
        self.num_node = self.data.num_node
        self.timecount = {}


    def run_exp(self):
        self.node_unlearning()

    def node_unlearning(self):
        self.logger.info("unlearning node number:{}".format(self.args["num_unlearned_nodes"]))
        # train
        self.features = self.preprocess_feature()
        self.NodeClassfier = NodeClassifier(self.args, self.data, self.model_zoo, self.logger)
        self.NodeClassfier.data.pre_features = self.features


        model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/" + \
                         self.args["base_model"]
        if os.path.exists(model_path):
            print(f"File {model_path} already exists.")
            self.NodeClassfier.load_model(model_path)
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'w'):
                print(f"File {model_path} created successfully.")
                self.NodeClassfier.train_GUIM_model()
                self.NodeClassfier.load_model(model_path)

        # select unlearning dataset
        train_nodes = np.array(self.data.train_indices)
        shuffle_num = torch.randperm(train_nodes.size)
        self.unlearning_nodes = train_nodes[shuffle_num][:self.args["num_unlearned_nodes"]]
        np.savetxt(unlearning_path, self.unlearning_nodes, fmt="%d")
        
        # self.unlearning_nodes = np.loadtxt(unlearning_path, dtype=int)
        F1_score, Accuracy, Recall = self.NodeClassfier.eval_unlearning(self.features, self.unlearning_nodes)
        self.logger.info(
            'Original Model Unlearning : F1_score = {}  Accuracy = {}  Recall = {}'.format(F1_score, Accuracy, Recall))
        self.unlearning_num = self.unlearning_nodes.size
        self.unlearning_mask = torch.zeros(self.num_node, dtype=torch.bool)
        self.unlearning_mask[self.unlearning_nodes] = True

        # compute influence_score
        start_time = time.time()
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            # self.softlabels = self.NodeClassfier.model.get_softlabel(self.features.cuda()).cpu().detach().numpy()
            # get frozen model param
            self.original_emb = self.NodeClassfier.model.get_embedding(self.features).clone().detach().float()
            self.original_softlabels = self.NodeClassfier.model.get_softlabel(self.features).clone().detach().float()
        else:
            # self.softlabels = self.NodeClassfier.model.get_softlabel(self.data.x,self.data.edge_index).cpu().detach().numpy()
            # get frozen model param
            self.original_emb = self.NodeClassfier.model.get_embedding(self.data.x,self.data.edge_index).clone().detach().float()
            self.original_softlabels = self.NodeClassfier.model.get_softlabel(self.data.x,self.data.edge_index).clone().detach().float()

        self.feature_similarity = self.cal_similarity(self.features, self.unlearning_nodes, True).cuda()
        print("feature_similarity ready!")
        self.label_similarity = self.cal_similarity(self.original_softlabels, self.unlearning_nodes, False).cuda()
        print("label_similarity ready!")
        self.influence_score = (self.feature_similarity + self.alpha * self.label_similarity).clone().detach()
        print("influence_score ready!")
        del self.feature_similarity
        del self.label_similarity
        # self.logger.info("influence_score: {}".format(self.influence_score))

        # select edit datasets
        # self.activated_nodes = np.loadtxt("./data/GUIM/activated_nodes.txt",dtype=int)
        self.activated_nodes = self.activate_nodes()
        self.timecount["activating_time"] = time.time()-start_time
        print("activated_nodes ready!")
        del self.influence_score
        self.activated_mask = torch.zeros(self.num_node, dtype=torch.bool)
        self.activated_mask[self.activated_nodes] = True
        # np.savetxt(root_path + "/data/GUIM/activated_nodes.txt", self.activated_nodes, fmt="%d")

        parameter_list = []
        for param_tensor in self.NodeClassfier.model.parameters():
            param_clone = param_tensor.clone().detach().float()
            parameter_list.append(param_clone)
        self.original_w = parameter_list
        print("parameter_list ready!")

        # get prototype embedding
        self.prototype_embedding = self.get_prototype()
        # pred = self.NodeClassfier.model.emb2softlable(self.prototype_embedding.to(self.device)).cpu().detach().numpy()
        # pred = np.argmax(pred, axis=1)
        # self.logger.info("proto_acc:{}".format(accuracy_score(y_true=np.arange(0, self.data.num_classes), y_pred=pred)))

        # contrastive learning sampling
        activated_label = np.argmax(self.original_softlabels[self.activated_nodes].cpu().numpy(), axis=1)
        start_time = time.time()
        self.pos_pair = self.pos_sampling(self.activated_nodes, activated_label, self.unlearning_nodes)
        self.neg_pair = self.neg_sampling(self.activated_nodes, activated_label, self.unlearning_nodes)
        self.timecount["sampling_time"] = time.time()-start_time

        self.NodeClassfier.GUIM_unlearning(self.original_softlabels,
                                           self.original_w,
                                           self.unlearning_nodes,
                                           self.activated_nodes,
                                           self.pos_pair,
                                           self.neg_pair,
                                           self.features,
                                           self.prototype_embedding,
                                           self.timecount)
        F1_score, Accuracy, Recall = self.NodeClassfier.eval_unlearning(self.features, self.unlearning_nodes)
        self.logger.info(
            'Original Model Unlearning : F1_score = {}  Accuracy = {}  Recall = {}'.format(F1_score, Accuracy, Recall))
        # retrain
        # self.retrain(self.unlearning_nodes, self.edge_mask)

        self.print_time()
        # mia attack test
        self.NodeClassfier.load_model(root_path + "/data/model/node_level/"+self.args["dataset_name"]+ "/" + self.args["base_model"]+"_unlearning_best.pt")
        self.mia_attack()
        self.logger.info("Budget: {}".format(self.Budget))


    def preprocess_feature(self,edge_mask=None):
        start_time = time.time()
        if self.args["base_model"] == "SGC":
            propogation = SGConv(self.data.num_features,self.data.num_classes,K=3,bias=False)
            features_pre = propogation.forward_GUIM(self.data.x,self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "S2GC":
            propogation = S2GConv(self.data.num_features, self.data.num_classes, K=3, bias=False)
            features_pre = propogation.forward_GUIM(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "SIGN":
            self.data.xs = torch.tensor([x.detach().numpy() for x in self.data.xs]).cuda()
            self.data.xs = self.data.xs.transpose(0,1)
            return self.data.xs
        elif self.args["base_model"] == "GCN":
            propogation = propogation = SGConv(self.data.num_features,self.data.num_classes,K=2,bias=False)
            features_pre = propogation.forward_GUIM(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "GAT":
            propogation = propogation = SGConv(self.data.num_features, self.data.num_classes, K=2, bias=False)
            features_pre = propogation.forward_GUIM(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "SAGE":
            propogation = propogation = SGConv(self.data.num_features, self.data.num_classes, K=2, bias=False)
            features_pre = propogation.forward_GUIM(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        elif self.args["base_model"] == "GIN":
            propogation = propogation = SGConv(self.data.num_features, self.data.num_classes, K=2, bias=False)
            features_pre = propogation.forward_GUIM(self.data.x, self.data.edge_index)
            return features_pre.cuda()
        timesum = time.time()-start_time
        self.timecount["process_feaure_time"] = timesum


    def cal_similarity(self,vectors,unleaning_nodes,isfeature):
        if self.args["base_model"] == "SIGN" and isfeature:
            vectors = self.NodeClassfier.model.get_features(vectors)

        if isinstance(vectors, np.ndarray):
            vectors = torch.FloatTensor(vectors)
        unlearning_vector = vectors[unleaning_nodes,:]
        unlearning_vector = F.normalize(unlearning_vector, p=2, dim=1).clone().detach().float()
        vectors = F.normalize(vectors, p=2, dim=1).clone().detach().float()
        similarity_matrix = self.compute_similarity_in_chunks(unlearning_vector, vectors, 10000)

        return similarity_matrix
    
    def compute_similarity_in_chunks(self,unlearning_vector, vectors, chunk_size):
        num_vectors = vectors.size(0)
        similarity_matrix = torch.zeros(num_vectors, device=unlearning_vector.device)

        # Iterate through the vectors in chunks
        for start in range(0, num_vectors, chunk_size):
            end = min(start + chunk_size, num_vectors)
            chunk = vectors[start:end]
            
            # Compute the chunk of the similarity matrix
            similarity_chunk = torch.matmul(unlearning_vector, chunk.T)
            
            # Sum the results along the dimension
            similarity_chunk_sum = torch.sum(similarity_chunk, dim=0)
            
            # Store the results in the corresponding part of the similarity matrix
            similarity_matrix[start:end] = similarity_chunk_sum

        return similarity_matrix

    def activate_nodes(self):
        activated_node  = list()
        total_influence = torch.zeros(self.num_node)
        self.influence_score[self.unlearning_nodes] = 0

        max_score,max_node,count = 0, 0, 0
        # topk_val,activated_node = torch.topk(total_influence,self.Budget)
        topk_val,activated_node = torch.topk(self.influence_score,self.Budget)
        return np.array(activated_node.cpu(),dtype=int)

    def pos_sampling(self, activated_nodes,activated_labels,unlearning_nodes):
        pos_pair = {}
        labels = self.data.y[self.data.train_mask].cpu().numpy()  # 提前将labels转为numpy数组
        unlearning_set = set(unlearning_nodes)  # 转为集合加速查找
        activated_set = set(activated_nodes)  # 转为集合加速查找

        for count, node in enumerate(activated_nodes):
            pos_pair[node] = []
            indice = np.where(labels == activated_labels[count])[0]  # 获取索引并直接使用数组
            np.random.shuffle(indice)  # 打乱索引
            number = 0
            embeddings = []
            for index in indice:
                if index not in unlearning_set and index not in activated_set:
                    embeddings.append(self.original_emb[index].cpu().numpy())
                    number += 1
                    if number >= 5:
                        break

            if not embeddings:
                pos_pair[node].append(np.zeros_like(self.original_emb[0].cpu()))
            else:
                pos_pair[node] = np.mean(embeddings, axis=0)
        pos_pair = {k: np.array(v) for k, v in pos_pair.items()}
            


        return pos_pair

    def neg_sampling(self, activated_nodes, activated_labels, unlearning_nodes):
        neg_pair = {}
        labels = self.data.y[activated_nodes].cpu().numpy()  # 提前将labels转为numpy数组
        unlearning_set = set(unlearning_nodes)  # 转为集合加速查找

        for count, node in enumerate(activated_nodes):
            neg_pair[node] = []
            embeddings = []

            # 添加当前节点的embedding 5次
            current_emb = self.original_emb[node].cpu().numpy()
            embeddings.extend([current_emb] * 5)

            # 获取符合条件的节点索引
            indice = np.where(labels == activated_labels[count])[0]
            np.random.shuffle(indice)  # 打乱索引

            number = 5
            for index in indice:
                if index not in unlearning_set:
                    embeddings.append(self.original_emb[index].cpu().numpy())
                    number += 1
                    if number >= 10:
                        break

            # 计算均值
            neg_pair[node] = np.mean(embeddings, axis=0)
            
        return neg_pair
    #5个自身向量，5个负样本

    def delete_node(self):
        self.edge_mask = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)
        for node in self.unlearning_nodes:
            self.edge_mask[self.data.edge_index[0] == node] = False
            self.edge_mask[self.data.edge_index[1] == node] = False

    def retrain(self, unlearning_nodes, edge_mask):
        self.NodeClassfier.data.edge_index = self.NodeClassfier.data.edge_index[:,edge_mask]
        self.NodeClassfier.train_model()
        edge_mask = True

    def mia_attack(self):

        self.mia_num = self.unlearning_num
        original_softlabels_member = self.original_softlabels[self.unlearning_nodes]
        original_softlabels_non = self.original_softlabels[self.data.test_indices[:self.mia_num]]
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args[
            "base_model"] == "SIGN":
            unlearning_softlabels_member = self.NodeClassfier.model.get_softlabel(self.features[self.unlearning_nodes])
            unlearning_softlabels_non = self.NodeClassfier.model.get_softlabel(
                self.features[self.data.test_indices[:self.mia_num]])
        else:
            unlearning_softlabels_member = self.NodeClassfier.model.get_softlabel(self.data.x,self.data.edge_index)[self.unlearning_nodes]
            unlearning_softlabels_non = self.NodeClassfier.model.get_softlabel(
                self.data.x,self.data.edge_index)[self.data.test_indices[:self.mia_num]]

        mia_test_y = torch.cat((torch.ones(self.mia_num), torch.zeros(self.mia_num)))
        posterior1 = torch.cat((original_softlabels_member, original_softlabels_non), 0).cpu().detach()
        posterior2 = torch.cat((unlearning_softlabels_member, unlearning_softlabels_non), 0).cpu().detach()
        posterior = np.array([np.linalg.norm(posterior1[i]-posterior2[i]) for i in range(len(posterior1))])
        # self.logger.info("posterior:{}".format(posterior))
        auc = roc_auc_score(mia_test_y, posterior.reshape(-1, 1))
        self.logger.info("auc:{}".format(auc))
        self.plot_auc(mia_test_y, posterior.reshape(-1, 1))



    def get_prototype(self):
        train_indices = torch.tensor(self.data.train_indices,requires_grad=False)
        prototype_mask = torch.ones_like(train_indices, dtype=torch.bool)
        u_nodes = torch.from_numpy(self.unlearning_nodes)
        a_nodes = torch.from_numpy(self.activated_nodes)
        prototype_mask[torch.isin(train_indices, torch.cat((u_nodes,a_nodes)))] = False
        prototype_indices = train_indices[prototype_mask]
        tmp_emb = self.original_emb[prototype_indices]
        tmp_y = self.data.y[prototype_indices].cpu()
        unique_labels = np.unique(tmp_y)
        # 初始化一个数组用于存储每个类别的平均嵌入向量
        class2embeddings = torch.zeros((self.data.num_classes, tmp_emb.shape[1]),requires_grad=False)


        # 遍历每个类别
        for label in unique_labels:
            # 获取属于当前类别的样本索引
            indices = np.where(tmp_y == label)[0]
            # 获取属于当前类别的嵌入向量
            class_emb = tmp_emb[indices]
            # 计算当前类别的平均嵌入向量
            class_avg_emb = torch.mean(class_emb, dim=0)
            class2embeddings[label.item()] = class_avg_emb

        return class2embeddings

    def plot_auc(self,y_true,y_score):
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



    def print_time(self):
        for key,value in self.timecount.items():
            if key == "training_time":
                value = value/self.args["unlearning_epochs"]
            self.logger.info("{} : {}s".format(key,value))
        
















