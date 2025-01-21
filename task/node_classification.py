import logging
import os
import random
import time
import copy
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score,recall_score
from torch_geometric.data import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from model.base_gnn.ceu_model import CEU_GNN
from utils import  utils
from torch.autograd import grad
import config
from tqdm import tqdm
from utils.utils import get_loss_fct, trange, Reverse_CE
from config import root_path


class NodeClassifier:
    def __init__(self,args,data,model_zoo,logger):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.data = data
        self.model_zoo = model_zoo
        self.model = model_zoo.model
        self.logger = logger
        self.model_name = self.args['base_model']

    def train_model(self,retrain=False):
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_zoo.model_config.lr, weight_decay=self.model_zoo.model_config.decay)
        time_sum = 0
        for epoch in range(self.args['num_epochs']):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            if self.args["base_model"] == "SIGN":
                out = self.model(self.data.xs)
            else:
                out = self.model(self.data.x,self.data.edge_index)
            loss = F.cross_entropy(out[self.data.train_mask],self.data.y[self.data.train_mask])
            
            loss.backward()
            self.optimizer.step()
            self.best_valid_acc = 0
            time_sum += time.time() - start_time

            if (epoch+1) % self.args["test_freq"] == 0:
                F1_score, Accuracy, Recall = self.evaluate_model()
                self.logger.info(
                    'epoch: {}  F1_score = {}  Accuracy = {}  Recall = {}'.format(epoch, F1_score, Accuracy, Recall))
                
                self.logger.info("epoch:{} loss:{}".format(epoch,loss))



                if self.args["unlearning_methods"] == 'GNNDelete':
                    self.GNNDelete_eval(epoch,retrain)
        self.logger.info("training_time:{}".format(time_sum/self.args['num_epochs']))

        #self.model.save_model(config.MODEL_PATH +self.args['dataset_name'] + "/" + self.args['base_model'])

    def train_GUIM_model(self,retrain=False):
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_zoo.model_config.lr, weight_decay=self.model_zoo.model_config.decay)
        start_time = time.time()
        best_acc = 0
        best_w = 0
        for epoch in range(self.args['num_epochs']):
            self.model.train()
            self.optimizer.zero_grad()
            if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
                out = self.model(self.data.pre_features[self.data.train_indices])
                loss = F.cross_entropy(out, self.data.y[self.data.train_indices])
            else:
                out = self.model(self.data.x,self.data.edge_index)
                loss = F.cross_entropy(out[self.data.train_indices], self.data.y[self.data.train_indices])

            self.logger.info("epoch:{} loss:{}".format(epoch,loss))
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % self.args["test_freq"] == 0:
                F1_score, Accuracy, Recall = self.evaluate_GUIM_model(self.data.pre_features[self.data.test_indices])
                self.logger.info(
                    'epoch: {}  F1_score = {}  Accuracy = {}  Recall = {}'.format(epoch, F1_score, Accuracy, Recall))
                if Accuracy > best_acc :
                    best_acc = Accuracy
                    best_w = copy.deepcopy(self.model.state_dict())
        if self.args["unlearn_task"] == "edge":
            model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/" + \
                         self.args["base_model"] + self.args["proportion_unlearned_edges"]
        else:
            model_path = root_path + "/data/model/" + self.args["unlearn_task"] + "_level/" + self.args["dataset_name"] + "/" + \
                         self.args["base_model"]
        self.save_model(model_path,best_w)
        self.logger.info("best:{}".format(best_acc))


    @torch.no_grad()
    def evaluate_GUIM_model(self,test_features):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            y_pred = self.model.get_softlabel(test_features).cpu()
        else:
            y_pred = self.model.get_softlabel(self.data.x,self.data.edge_index).cpu()
            y_pred = y_pred[self.data.test_indices]
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        F1_score = f1_score(y[self.data.test_indices], y_pred, average="micro")
        Acc_score = accuracy_score(y_true=y[self.data.test_indices], y_pred=y_pred)
        Recall_score = recall_score(y_true=y[self.data.test_indices], y_pred=y_pred, average="micro")
        return F1_score, Acc_score, Recall_score

    def GUIM_unlearning(self,
                        original_softlabels,
                        original_w,
                        unlearning_nodes,
                        activated_nodes,
                        pos_pair,
                        neg_pair,
                        original_feaures,
                        prototype_embedding,
                        timecount):
        self.model.train()
        timecount["training_time"] = 0
        self.lam = 1
        self.tau = 0.5
        self.para1 = 1
        self.para2 = 0.005
        self.para3 = 1
        self.para4 = 0.5
        self.para5 = 10
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        prototype_embedding = prototype_embedding.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_zoo.model_config.lr,
                                          weight_decay=self.model_zoo.model_config.decay)

        indices = torch.randperm(self.data.y[unlearning_nodes].size(0))

        # 使用这些索引重新排列原始张量
        random_class = self.data.y[unlearning_nodes][indices]
        random_emb_proto = prototype_embedding[random_class]
        average_probs = torch.ones(self.data.num_classes) / self.data.num_classes
        stacked_probs = average_probs.repeat(self.args["num_unlearned_nodes"], 1).detach()
        best = 0
        best_w = 0
        F1_score = 0
        for epoch in range(self.args['unlearning_epochs']):
            self.model.train()
            self.optimizer.zero_grad()
            if (epoch+1) % self.args["test_freq"] == 0:
                F1_score,Accuracy,Recall = self.evaluate_GUIM_model(original_feaures[self.data.test_indices])
                self.logger.info('epoch: {}  F1_score = {}  Accuracy = {}  Recall = {}'.format(epoch, F1_score,Accuracy,Recall))
            ###get emb softlabel###
            # softlabels_U = F.softmax(self.model(original_feaures[unlearning_nodes]),dim=1)
            # # softlabels_U = self.model.get_softlabel(original_feaures[unlearning_nodes])
            # softlabels_E = F.softmax(self.model(features_unlearning[activated_nodes]),dim=1)
            # embedding_E = self.model.get_embedding(features_unlearning[activated_nodes])
            # embedding_U = self.model.get_embedding(original_feaures[unlearning_nodes])

            ## 3.25 19-16
            # softlabels_U = F.softmax(self.model(original_feaures[unlearning_nodes]),dim=1)
            start_time = time.time()
            if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args[
                "base_model"] == "SIGN":
                softlabels_U = self.model.get_softlabel(original_feaures[unlearning_nodes])
                softlabels_E = F.softmax(self.model(original_feaures[activated_nodes]),dim=1)
                embedding_E = self.model.get_embedding(original_feaures[activated_nodes])
                embedding_U = self.model.get_embedding(original_feaures[unlearning_nodes])
            else:
                softlabels_U = self.model.get_softlabel(self.data.x,self.data.edge_index)[unlearning_nodes]
                softlabels_E = self.model.get_softlabel(self.data.x,self.data.edge_index)[activated_nodes]
                embedding_E = self.model.get_embedding(self.data.x,self.data.edge_index)[activated_nodes]
                embedding_U = self.model.get_embedding(self.data.x,self.data.edge_index)[unlearning_nodes]
            w_U = []
            ### cal loss###
            loss_w = 0
            for param_tensor in self.model.parameters():
                param_clone = param_tensor
                w_U.append(param_tensor)
            for i in range(len(original_w)):
                delta_w = (w_U[i] - original_w[i])
                loss_w += torch.norm(delta_w)
            creterion_MSE = torch.nn.MSELoss(reduction="mean")
            # loss_pE = creterion_MSE(softlabels_E,original_softlabels[activated_nodes])
            smooth_factor = 1e-9
            loss_pE = F.kl_div(torch.log(softlabels_E + smooth_factor),original_softlabels[activated_nodes] + smooth_factor, reduction='batchmean')
            pos_tensors = [torch.tensor(pos_pair[node]) for node in activated_nodes]
            positive_tensors = torch.stack([F.normalize(pos_tensor, p=2, dim=0) for pos_tensor in pos_tensors], dim=0).cuda()
            pos_scores = torch.exp(torch.einsum('ij,ij->i', F.normalize(embedding_E, p=2, dim=1),
                                                positive_tensors )/ self.tau)
            neg_scores = torch.zeros_like(pos_scores)

            ####6/1#########
            neg_tensors = [torch.tensor(neg_pair[node]) for node in activated_nodes]
            negtive_tensors = torch.stack([F.normalize(negtive_tensor, p=2, dim=0) for negtive_tensor in neg_tensors],
                                           dim=0).cuda()
            neg_scores = torch.exp(torch.einsum('ij,ij->i', F.normalize(embedding_E, p=2, dim=1),
                                                negtive_tensors) / self.tau)

            ################
            # for i, node in enumerate(activated_nodes):
            #     # neg_tensors = F.normalize(torch.tensor(neg_pair[node][0]), p=2, dim=0).cuda()
            #     # tmp_emb = F.normalize(embedding_E[i], p=2, dim=0)
            #     # neg_scores = 10 * torch.dot(neg_tensors,tmp_emb)
            #
            #     neg_tensors = [torch.tensor(neg_emb) for neg_emb in neg_pair[node]]
            #     negtive_tensors = torch.stack([F.normalize(neg_tensor, p=2, dim=0) for neg_tensor in neg_tensors], dim=0).cuda()
            #     neg_scores[i] = torch.exp(torch.einsum('d,nd->n', F.normalize(embedding_E[i], p=2, dim=0),
            #                      negtive_tensors) / self.tau).sum()
            loss_hE = -torch.log(pos_scores / 50 * neg_scores).sum()
             


            # embedding_E_proto = prototype_embedding[self.data.y[activated_nodes]]
            # loss_proto = creterion_MSE(embedding_E,embedding_E_proto)
            # loss_proto = F.kl_div(torch.log(embedding_E),embedding_E_proto, reduction='batchmean')
            # for node in activated_nodes:
            #     emb = embedding_E[node]
            #     pos_num = torch.exp(torch.mul(F.normalize(emb, p=2, dim=0),F.normalize(torch.from_numpy(pos_pair[node]).to(self.device), p=2, dim=0)).sum()/self.tau)
            #     neg_num = 0
            #     for neg_emb in neg_pair[node]:
            #         neg_num += torch.exp(torch.mul(F.normalize(emb, p=2, dim=0),F.normalize(torch.from_numpy(neg_emb).to(self.device), p=2, dim=0)).sum()/self.tau)
            #     loss_hE += -torch.log(pos_num/neg_num)

            #尝试用随机标签的方式遗忘
            loss_pU = F.cross_entropy(softlabels_U, random_class.to(self.device))

            #尝试用平均概率的方式遗忘
            # loss_pU = F.kl_div(torch.log(softlabels_U+smooth_factor),stacked_probs.to(self.device), reduction='batchmean')
            loss_emb_U = creterion_MSE(embedding_U, random_emb_proto)
            # loss_emb_U = 0
            # onehot = F.one_hot(self.data.y[unlearning_nodes]).detach()

            loss = self.lam * ( self.para1 * loss_w +  self.para3 * loss_pE ) + (self.para2 * loss_hE +  self.para4 * (loss_pU+ loss_emb_U)  )
            # loss = self.lam * ( self.para3 * loss_pE ) + (self.para2 * loss_hE +  self.para4 * (loss_pU+ loss_emb_U)  )
            # loss = self.lam * (self.para1 * loss_w + self.para3 * loss_pE + self.para5 * loss_proto) + (
            #             self.para2 * loss_hE + self.para4 * (loss_pU + loss_emb_U))

            self.logger.info("loss_w: {}  loss_hE: {} loss_pE: {} loss_pU: {} ".format(self.para1*loss_w,
                                                                                                    self.para2 * loss_hE,
                                                                                                    self.para3 * loss_pE,
                                                                                                    self.para4* (loss_pU+ loss_emb_U)))
            self.logger.info("epoch:{} loss:{}".format(epoch, loss))
            loss.backward()
            self.optimizer.step()
            timecount["training_time"] += time.time() - start_time

            ##test##
            # if (epoch+1) % self.args["test_freq"] == 0:
            #     F1_score,Accuracy,Recall = self.evaluate_GUIM_model(original_feaures[self.data.test_indices])
            #     self.logger.info('epoch: {}  F1_score = {}  Accuracy = {}  Recall = {}'.format(epoch, F1_score,Accuracy,Recall))
            if F1_score > best and epoch > 30:
                best = F1_score
                best_w = copy.deepcopy(self.model.state_dict())
                F1_score,Accuracy,Recall = self.eval_unlearning(original_feaures,unlearning_nodes)
                self.logger.info('Unlearning: F1_score = {}  Accuracy = {}  Recall = {}'.format(F1_score,Accuracy,Recall))
        F1_score, Accuracy, Recall = self.eval_unlearning(original_feaures,unlearning_nodes)
        self.logger.info("para1:{}   para2:{}   para3:{}   para4:{}   para5:{}".format(self.para1,
                                                                                       self.para2,
                                                                                       self.para3,
                                                                                       self.para4,
                                                                                       self.para5))
        self.logger.info('best: {}'.format(best))
        save_path = root_path + "/data/model/node_level/"+self.args["dataset_name"]+ "/" + self.args["base_model"]+"_unlearning_best.pt"
        with open(save_path,'w') as file:
            self.save_model(save_path,best_w)




    @torch.no_grad()
    def eval_unlearning(self,features,unlearning_nodes,edge_mask=None):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            y_pred = self.model.get_softlabel(features[unlearning_nodes].cuda()).cpu()
        else:
            y_pred = self.model.get_softlabel(self.data.x,self.data.edge_index).cpu()
            y_pred = y_pred[unlearning_nodes]
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        F1_score = f1_score(y[unlearning_nodes], y_pred, average="micro")
        Acc_score = accuracy_score(y_true=y[unlearning_nodes], y_pred=y_pred)
        Recall_score = recall_score(y_true=y[unlearning_nodes], y_pred=y_pred,average="micro")
        return F1_score,Acc_score,Recall_score


    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SIGN":
            y_pred = self.model(self.data.xs).cpu()
        else:
            y_pred = self.model(self.data.x,self.data.edge_index).cpu()
        y = self.data.y.cpu()
        y_pred = np.argmax(y_pred, axis=1)
        F1_score = f1_score(y[self.data.test_mask.cpu()], y_pred[self.data.test_mask.cpu()], average="micro")
        Acc_score = accuracy_score(y_true=y[self.data.test_mask.cpu()], y_pred=y_pred[self.data.test_mask.cpu()])
        Recall_score = recall_score(y_true=y[self.data.test_mask.cpu()], y_pred=y_pred[self.data.test_mask.cpu()],average="micro")
        return F1_score,Acc_score,Recall_score
    



    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()

        if self.device == 'cpu':
            model = model.to('cpu')

        # if hasattr(data, 'dtrain_mask'):
        #     mask = data.dtrain_mask
        # else:
        #     mask = data.dr_mask
        #z = F.log_softmax(model(data.x, data.edge_index), dim=1)

        # DT AUC AUP
        if self.args["base_model"] == "SGC" or self.args["base_model"] == "S2GC" or self.args["base_model"] == "SIGN":
            z = self.model(self.data.features_pre)
        else:
            z = self.model(self.data.x, self.data.edge_index)
        loss = F.cross_entropy(z[data.val_mask], data.y[data.val_mask]).cpu().item()
        pred = torch.argmax(z[data.val_mask], dim=1).cpu()
        true_lable = data.y[data.val_mask]
        dt_acc = accuracy_score(data.y[data.val_mask].cpu(), pred)
        recall = recall_score(data.y[data.val_mask].cpu(), pred,average='micro')
        dt_f1 = f1_score(data.y[data.val_mask].cpu(), pred, average='micro')

        # DF AUC AUP
        # if self.args.unlearning_model in ['original', 'original_node']:
        #     df_logit = []
        # else:
        #     df_logit = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        # if len(df_logit) > 0:
        #     df_auc = []
        #     df_aup = []

        #     # Sample pos samples
        #     if len(self.df_pos_edge) == 0:
        #         for i in range(500):
        #             mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
        #             idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
        #             mask[idx] = True
        #             self.df_pos_edge.append(mask)

        #     # Use cached pos samples
        #     for mask in self.df_pos_edge:
        #         pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()

        #         logit = df_logit + pos_logit
        #         label = [0] * len(df_logit) +  [1] * len(df_logit)
        #         df_auc.append(roc_auc_score(label, logit))
        #         df_aup.append(average_precision_score(label, logit))

        #     df_auc = np.mean(df_auc)
        #     df_aup = np.mean(df_aup)

        # else:
        #     df_auc = np.nan
        #     df_aup = np.nan

        # Logits for all node pairs
        # if pred_all:
        #     logit_all_pair = (z @ z.t()).cpu()
        # else:
        #     logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_acc': dt_acc,
            f'{stage}_dt_f1': dt_f1,
        }

        if self.device == 'cpu':
            model = model.to(self.device)

        return loss, dt_acc,recall, dt_f1, log

    @torch.no_grad()
    def prediction_info(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        y_pred = self.model(self.data.x, self.data.edge_index).cpu()
        y = self.data.y.cpu()
        return y_pred[self.data.test_mask.cpu()], y[self.data.test_mask.cpu()]



    def posterior(self):
        self.logger.debug("generating posteriors")
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.model.eval()

        self._gen_test_loader()
        if self.model_name in ['GCN',"SGC","S2GC"] and self.args["unlearning_methods"] == "GIF":
            posteriors = self.model.GIF_inference(self.data.x, self.test_loader, self.edge_weight, self.device)
        elif self.model_name in ["SIGN"] and self.args["unlearning_methods"] == "GIF":
            posteriors = self.model.GIF_inference(self.data)
        else:
            posteriors = self.model.inference(self.data.x, self.test_loader, self.device)

        for _, mask in self.data('test_mask'):
            posteriors = F.log_softmax(posteriors[mask.cpu()], dim=-1)

        return posteriors.detach()

    def posterior_other(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        if self.args["base_model"] == "SIGN":
            posteriors = self.model(self.data.xs)
        else:
            posteriors = self.model(self.data.x,self.data.edge_index)
        for _, mask in self.data('test_mask'):
            posteriors = posteriors[mask]

        return posteriors.detach()

    def generate_embeddings(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_test_loader()

        if self.model_name == 'GCN':
            logits = self.model.inference(self.data.x, self.test_loader, self.edge_weight, self.device)
        else:
            logits = self.model.inference(self.data.x, self.test_loader, self.device)
        return logits


    def _gen_test_loader(self):
        test_indices = np.nonzero(self.data.test_mask.cpu().numpy())[0]

        if not self.args['use_test_neighbors']:
            edge_index = utils.filter_edge_index(self.data.edge_index, test_indices, reindex=False)
        else:
            edge_index = self.data.edge_index

        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[1, 3], [3, 1]])

        self.test_loader = NeighborSampler(
            edge_index, node_idx=None,
            sizes=[-1], num_nodes=self.data.num_nodes,
            # sizes=[5], num_nodes=self.data.num_nodes,
            batch_size=self.args['test_batch_size'], shuffle=False,
            num_workers=0)
        # self.test_loader = NeighborSampler(
        #     edge_index, node_idx=None,
        #     sizes=[-1], num_nodes=self.data.num_nodes,
        #     # sizes=[5], num_nodes=self.data.num_nodes,
        #     batch_size=self.args['test_batch_size'], shuffle=False,
        #     num_workers=0)

        if self.model_name == 'GCN':
            _, self.edge_weight = gcn_norm(self.data.edge_index, edge_weight=None, num_nodes=self.data.x.shape[0],
                                           add_self_loops=False)
    def _gen_train_loader(self):
        self.logger.info("generate train loader")
        train_indices = np.nonzero(self.data.train_mask.cpu().numpy())[0]
        edge_index = utils.filter_edge_index(self.data.edge_index, train_indices, reindex=False)
        if edge_index.shape[1] == 0:
            edge_index = torch.tensor([[1, 2], [2, 1]])

        self.train_loader = NeighborSampler(
            edge_index, node_idx=self.data.train_mask,
            sizes=[5, 5], num_nodes=self.data.num_nodes,
            batch_size=self.args['batch_size'], shuffle=True,
            num_workers=0)

        if self.model_name in ['GCN','SGC','S2GC']:
            _, self.edge_weight = gcn_norm(self.data.edge_index, edge_weight=None, num_nodes=self.data.x.shape[0],
                                           add_self_loops=False)

            if self.args["GIF_method"] in ["GIF", "IF"]:
                _, self.edge_weight_unlearn = gcn_norm(
                    self.data.edge_index_unlearn,
                    edge_weight=None,
                    num_nodes=self.data.x.shape[0],
                    add_self_loops=False)

        self.logger.info("generate train loader finish")

    @torch.no_grad()
    def verification_error(self,model1, model2):
        '''L2 distance between aproximate model and re-trained model'''

        model1 = model1.to('cpu')
        model2 = model2.to('cpu')

        modules1 = {n: p for n, p in model1.named_parameters()}
        modules2 = {n: p for n, p in model2.named_parameters()}

        all_names = set(modules1.keys()) & set(modules2.keys())

        print(all_names)

        diff = torch.tensor(0.0).float()
        for n in all_names:
            diff += torch.norm(modules1[n] - modules2[n])

        return diff

    def save_model(self, save_path,model_dict=None):
        with open(save_path,mode='w') as file:
            if model_dict is not None:
                self.logger.info('saving best model {}'.format(save_path))
                torch.save(model_dict, save_path)
            else:
                self.logger.info('saving model {}'.format(save_path))
                torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path):
        self.logger.info('loading model {}'.format(save_path))
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(save_path, map_location=device))



