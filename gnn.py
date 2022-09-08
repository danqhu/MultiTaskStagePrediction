import math
import numpy as np
from turtle import forward
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from dgl.data import DGLDataset
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
import config
import time
import copy
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from torch.utils.tensorboard import SummaryWriter
from preprocessing4graph import construct_PosNeg_graphs, construct_PosNeg_graphs_v2, construct_PosNeg_graphs_v3, construct_PosNeg_graphs_v4
from matplotlib import pyplot as plt


# class GNNLayer(Module):
#     def __init__(self, in_features, out_features):
#         super(GNNLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         torch.nn.init.xavier_uniform_(self.weight)

#     def forward(self, features, adj, active=True):
#         support = torch.mm(features, self.weight)
#         # output = torch.spmm(adj, support)
#         output = torch.mm(adj, support)
#         if active:
#             output = F.relu(output)
#         return output



# class GraphAttentionLayer(nn.Module):

#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """

#     def __init__(self, in_features, out_features, alpha=0.2):
#         super(GraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha

#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)

#         self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
#         nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

#         self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
#         nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def forward(self, input, adj, M, concat=True):
#         h = torch.mm(input, self.W)
#         #前馈神经网络
#         attn_for_self = torch.mm(h,self.a_self)       #(N,1)
#         attn_for_neighs = torch.mm(h,self.a_neighs)   #(N,1)
#         attn_dense = attn_for_self + torch.transpose(attn_for_neighs,0,1)
#         attn_dense = torch.mul(attn_dense,M)
#         attn_dense = self.leakyrelu(attn_dense)            #(N,N)

#         #掩码（邻接矩阵掩码）
#         zero_vec = -9e15*torch.ones_like(adj)
#         adj = torch.where(adj > 0, attn_dense, zero_vec)
#         attention = F.softmax(adj, dim=1)
#         h_prime = torch.matmul(attention,h)

#         if concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime

#     def __repr__(self):
#             return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphDataset(DGLDataset):
    def __init__(self, nodes_data, edges_data, labels, train_mask=None, val_mask=None, test_mask=None, edges_weights=None):
        self.nodes_data = nodes_data
        self.edges_data = edges_data
        self.labels = labels
        self.num_nodes = nodes_data.shape[0]
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.train_mask[train_mask] = True
        self.val_mask[val_mask] = True
        self.test_mask[test_mask] = True
        self.edges_weights = edges_weights
        super().__init__(name='similar_patient_net')
    

    def process(self):

        nodes_features = self.nodes_data
        nodes_labels = self.labels
        edges_src = self.edges_data[:,0]
        edges_dst = self.edges_data[:,1]
        

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=self.nodes_data.shape[0])
        self.graph.ndata['feat'] = torch.tensor(nodes_features, dtype=torch.float32)
        self.graph.ndata['label'] = torch.tensor(nodes_labels,dtype=torch.long)

        self.graph.ndata['train_mask'] = self.train_mask
        self.graph.ndata['val_mask'] = self.val_mask
        self.graph.ndata['test_mask'] = self.test_mask

        if self.edges_weights is not None:
            self.graph.edata['edges_weights'] = torch.tensor(self.edges_weights, dtype=torch.float32)



    def __getitem__(self, idx):
        
        return self.graph
    

    def __len__(self):
        return 1




class gcn(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()

        self.conv1 = GraphConv(in_feats, hid_feats[0])
        # self.conv2 = GraphConv(hid_feats[0], hid_feats[1])
        # self.conv3 = GraphConv(hid_feats[1], hid_feats[2])
        self.fc = nn.Linear(hid_feats[-1], out_feats)


    def forward(self, graph, inputs, edges_weights):
        h = self.conv1(graph, inputs, edge_weight=edges_weights)
        h = F.relu(h)
        # h = self.conv2(graph, h, edge_weight=edges_weights)
        # h = F.relu(h)
        # h = self.conv3(graph, h, edge_weight=edges_weights)
        # h = F.relu(h)
        h = self.fc(h)

        return h

class gat(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_heads=2) -> None:
        super().__init__()

        self.conv1 = GATConv(in_feats, hid_feats[0], num_heads)
        # self.conv2 = GATConv(hid_feats[0]*num_heads, hid_feats[1], num_heads)
        # self.conv3 = GATConv(hid_feats[1]*num_heads, hid_feats[2], num_heads)
        self.fc = nn.Linear(hid_feats[0], out_feats)
    
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        # h = torch.flatten(h, start_dim=1)
        # h = self.conv2(graph, h)
        # h = F.relu(h)
        # h = torch.flatten(h, start_dim=1)
        # h = self.conv3(graph, h)
        # h = F.relu(h)
        h = torch.mean(h, dim=1)
        h = self.fc(h)

        return h


def train_gnn(
    hyper_params,
    model,  
    nodes_data,
    edges_data,
    labels,
    train_mask,
    val_mask,
    test_mask,
    best_model_dir,
    fold,
    cv_idx,
    write_log=False
):
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model.to(device)


    graph = GraphDataset(nodes_data, edges_data, labels, train_mask, val_mask, test_mask)[0].to(device)
    # graph = dgl.add_self_loop(graph)

    features = graph.ndata['feat']
    


    best_model_wts = copy.deepcopy(model.state_dict())
    lr = hyper_params['learning_rate']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    
    criterian = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    # optimizer = torch.optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999],weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5)

    model.train()
    total_loss = 0
    best_val_loss = float("inf")
    best_val_auc = float("-inf")
    best_val_ap = float("-inf")
    
    # writer = SummaryWriter(log_dir='./runs/')
    
    
    for epoch in range(num_epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # train_sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()

        optimizer.zero_grad()

        outputs = model(graph, features)

        loss = criterian(outputs[graph.ndata['train_mask']], graph.ndata['label'][graph.ndata['train_mask']])


        probas = softmax(outputs)
        total_loss += loss.item()
        
        

        

        # print('Epoch {} rank {} total loss {}'.format(epoch, rank, total_loss))

        
        loss.backward()
        optimizer.step()
        scheduler.step()

        y_probs = softmax(outputs[train_mask]).detach().cpu().numpy()[:, 1]
        y_true = labels[train_mask]



        train_auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        train_ap = average_precision_score(y_true=y_true, y_score=y_probs)
  
    
        # train_loss, train_auc, train_ap = evaluate_rnt(model, criterian, data_train, labels_train, data_image_filename_train)
        val_loss, val_auc, val_ap, _, _ = evaluate_gnn(model, criterian, nodes_data, edges_data, labels, train_mask, val_mask, test_mask, val_mask)
        test_loss, test_auc, test_ap, _, _ = evaluate_gnn(model, criterian, nodes_data, edges_data, labels, train_mask, val_mask, test_mask, test_mask, test=True)
        
        # if write_log:
        #     writer.add_scalars('Loss/fold'+str(fold), {'train':total_loss, 'val': val_loss, 'test':test_loss}, epoch)
        #     writer.add_scalars('AUC/fold'+str(fold), {'train':train_auc, 'val': val_auc, 'test':test_auc}, epoch)
        #     writer.add_scalars('AP/fold'+str(fold), {'train':train_ap, 'val': val_ap, 'test':test_ap}, epoch)


        
        print('Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} |'
                'valid AUC {:.3f}  | valid AP {:.3f} | test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}'.format(epoch, (time.time() - epoch_start_time), total_loss, val_loss, val_auc, val_ap, test_loss, test_auc, test_ap))
        print('-' * 89)

        total_loss = 0


        if epoch > 10 and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())

            path = best_model_dir + 'best_val_model_' + str(fold) + '.' + str(cv_idx) + '.pt'
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            
            torch.save({
                'model_state_dict': best_model_wts,
            }, path)



    # writer.close()
    return val_auc, val_ap, best_model_wts



def evaluate_gnn(
    eval_model,
    criterian,
    nodes_data,
    edges_data,
    labels,
    train_mask,
    val_mask,
    test_mask,
    mask,
    test=False 
):
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    graph = GraphDataset(nodes_data, edges_data, labels, train_mask, val_mask, test_mask)[0].to(device)
    # graph = dgl.add_self_loop(graph)
    
    features = graph.ndata['feat']

    labels = torch.tensor(labels, dtype=torch.long).to(device)

    mask_bool = graph.ndata['test_mask'] if test else graph.ndata['val_mask']

    # graph = GraphDataset(nodes_data, edges_data, labels)[0].to(device)
    # mask_bool = torch.zeros(nodes_data.shape[0], dtype=torch.bool)
    # mask_bool[mask] = True

    
    
    eval_model.to(device)

    softmax = nn.Softmax(dim=1)

    eval_model.eval()

    outputs = eval_model(graph, features)
    # outputs= outputs.view(-1)
    loss = criterian(outputs[mask_bool], labels[mask_bool])
    # probas = sigmoid(outputs)
    probas = softmax(outputs[mask_bool])


    y_probs = probas.detach().cpu().numpy()[:, 1]
    y_true = labels[mask].detach().cpu().numpy()
    total_loss = loss.item()


    auc = roc_auc_score(y_true=y_true, y_score=y_probs)
    ap = average_precision_score(y_true=y_true, y_score=y_probs)



    


    return total_loss, auc, ap, y_probs, y_true




def train_PosNeg_ps_graph(
    hyper_params,
    model,  
    data,
    labels,
    train_index,
    val_index,
    train_val_index,
    test_index,
    best_model_dir,
    fold,
    cv_idx,
    write_log=False
):
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model.to(device)

    train_val_data = data.loc[train_val_index, :].reset_index(inplace=False, drop=True)
    train_val_labels = labels[train_val_index]
    train_val_edges, _ = construct_PosNeg_graphs(data, labels, train_val_index, train_index, val_index, topk=3)


    train_val_graph = GraphDataset(train_val_data.values, train_val_edges, train_val_labels, train_index, val_index)[0].to(device)
    # graph = dgl.add_self_loop(graph)

    
    
    
    

    test_data = data.loc[test_index, :].reset_index(inplace=False, drop=True)
    test_labels = labels[test_index]

    train_val_edges, test_graph_pair = construct_PosNeg_graphs(data, labels, train_val_index, train_index, val_index, test_index, topk=3, test=True)
    
    test_graphs = []
        
    ''''''
    # 遍历每一个test sample
    # 将其与train_val samples 构建两个图，一个是与pos相连，一个是与neg相连
    # 将构建的图经过网络，预测test sample的属于哪一类，对两个图的结果进行softmax作为其预测结果
    for idx in range(test_data.shape[0]):
        edges_pos_graph, edges_neg_graph, test_index_graph = test_graph_pair[idx]
        train_val_test_data = np.concatenate((train_val_data.values, test_data.loc[idx,:].values[np.newaxis,:]), axis=0) 
        train_val_test_labesl = np.append(train_val_labels,test_labels[idx])
        test_graph_pos = GraphDataset(train_val_test_data, edges_pos_graph, train_val_test_labesl, train_index, val_index, test_index_graph)[0].to(device)
        test_graph_neg = GraphDataset(train_val_test_data, edges_neg_graph, train_val_test_labesl, train_index, val_index, test_index_graph)[0].to(device)
        test_graphs.append([test_graph_pos, test_graph_neg])

    
    
    
    
    
    
    
    
    
    
    features = train_val_graph.ndata['feat']
    


    best_model_wts = copy.deepcopy(model.state_dict())
    lr = hyper_params['learning_rate']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    
    criterian = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    # optimizer = torch.optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999],weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5)

    model.train()
    total_loss = 0
    best_val_loss = float("inf")
    best_val_auc = float("-inf")
    best_val_ap = float("-inf")
    
    # writer = SummaryWriter(log_dir='./runs/')
    

    for epoch in range(num_epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # train_sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()

        optimizer.zero_grad()

        outputs = model(train_val_graph, features)

        loss = criterian(outputs[train_val_graph.ndata['train_mask']], train_val_graph.ndata['label'][train_val_graph.ndata['train_mask']])


        probas = softmax(outputs[train_val_graph.ndata['train_mask']])
        total_loss += loss.item()
        
        

        

        # print('Epoch {} rank {} total loss {}'.format(epoch, rank, total_loss))

        
        loss.backward()
        optimizer.step()
        scheduler.step()

        y_probs = probas.detach().cpu().numpy()[:, 1]
        y_true = train_val_graph.ndata['label'][train_val_graph.ndata['train_mask']].detach().cpu().numpy()



        train_auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        train_ap = average_precision_score(y_true=y_true, y_score=y_probs)

        if epoch % 5 == 0:

            # train_loss, train_auc, train_ap = evaluate_rnt(model, criterian, data_train, labels_train, data_image_filename_train)
            val_loss, val_auc, val_ap, _, _ = evaluate_PosNeg_ps_graph(
                model, 
                criterian, 
                data, 
                labels, 
                train_index, 
                val_index,
                train_val_index,
                test_index,
                train_val_graph=train_val_graph)
                    
        
            test_loss, test_auc, test_ap, _, _ = evaluate_PosNeg_ps_graph(
                model, 
                criterian, 
                data, 
                labels, 
                train_index, 
                val_index,
                train_val_index,
                test_index,
                test=True,
                test_graphs=test_graphs)
        
            
            # if write_log:
            #     writer.add_scalars('Loss/fold'+str(fold), {'train':total_loss, 'val': val_loss, 'test':test_loss}, epoch)
            #     writer.add_scalars('AUC/fold'+str(fold), {'train':train_auc, 'val': val_auc, 'test':test_auc}, epoch)
            #     writer.add_scalars('AP/fold'+str(fold), {'train':train_ap, 'val': val_ap, 'test':test_ap}, epoch)


            
            print('Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} |'
                    'valid AUC {:.3f}  | valid AP {:.3f} | test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}'.format(epoch, (time.time() - epoch_start_time), total_loss, val_loss, val_auc, val_ap, test_loss, test_auc, test_ap))
            print('-' * 89)

            total_loss = 0


            if epoch > 10 and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_wts = copy.deepcopy(model.state_dict())

                path = best_model_dir + 'best_val_model_' + str(fold) + '.' + str(cv_idx) + '.pt'
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                
                torch.save({
                    'model_state_dict': best_model_wts,
                }, path)



    # writer.close()
    return val_auc, val_ap, best_model_wts



def evaluate_PosNeg_ps_graph(
    eval_model,
    criterian,
    data,
    labels,
    train_index=None,
    val_index=None,
    train_val_index=None,
    test_index=None,
    test=False,
    train_val_graph=None,
    test_graphs=None 
):
    if not test:
        device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
        
        
        
        eval_model.to(device)

        softmax = nn.Softmax(dim=1)

        eval_model.eval()

        outputs = eval_model(train_val_graph, train_val_graph.ndata['feat'])
        # outputs= outputs.view(-1)
        loss = criterian(outputs[train_val_graph.ndata['val_mask']], train_val_graph.ndata['label'][train_val_graph.ndata['val_mask']])
        # probas = sigmoid(outputs)
        probas = softmax(outputs[train_val_graph.ndata['val_mask']])


        y_probs = probas.detach().cpu().numpy()[:, 1]
        y_true = train_val_graph.ndata['label'][train_val_graph.ndata['val_mask']].detach().cpu().numpy()
        total_loss = loss.item()


        auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        ap = average_precision_score(y_true=y_true, y_score=y_probs)
    
    if test:
        device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")

       
        
        
        eval_model.to(device)

        softmax = nn.Softmax(dim=1)

        eval_model.eval()

        y_probs = np.zeros((len(test_index),2))
        y_true = np.zeros((len(test_index)))
        total_loss = 0

        for i, (pos_graph, neg_graph) in enumerate(test_graphs):

            pos_outputs = eval_model(pos_graph, pos_graph.ndata['feat'])
            # outputs= outputs.view(-1)
            pos_loss = criterian(pos_outputs[pos_graph.ndata['test_mask']], pos_graph.ndata['label'][pos_graph.ndata['test_mask']])
            # probas = sigmoid(outputs)
            pos_probas = softmax(pos_outputs[pos_graph.ndata['test_mask']])

            neg_outputs = eval_model(neg_graph, neg_graph.ndata['feat'])
            # outputs= outputs.view(-1)
            neg_loss = criterian(neg_outputs[neg_graph.ndata['test_mask']], neg_graph.ndata['label'][neg_graph.ndata['test_mask']])
            # probas = sigmoid(outputs)
            neg_probas = softmax(neg_outputs[neg_graph.ndata['test_mask']])


            
            y_probs[i,:] = (pos_probas.detach().cpu().numpy()+ neg_probas.detach().cpu().numpy())/2
            y_true[i] = pos_graph.ndata['label'][pos_graph.ndata['test_mask']].detach().cpu().numpy()
            # total_loss = loss.item()

        y_probs = y_probs[:,1]
        auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        ap = average_precision_score(y_true=y_true, y_score=y_probs)



    


    return total_loss, auc, ap, y_probs, y_true




def train_PosNeg_ps_graph_v2(
    hyper_params,
    model,  
    data,
    labels,
    train_index,
    val_index,
    train_val_index,
    test_index,
    best_model_dir,
    fold,
    cv_idx,
    write_log=False
):
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model.to(device)

    train_val_data = data.loc[train_val_index, :].reset_index(inplace=False, drop=True)
    train_val_labels = labels[train_val_index]
    train_data = train_val_data.loc[train_index, :].reset_index(inplace=False, drop=True)
    train_labels = train_val_labels[train_index]
    val_data = train_val_data.loc[val_index, :].reset_index(inplace=False, drop=True)
    val_labels = train_val_labels[val_index]

    test_data = data.loc[test_index, :].reset_index(inplace=False, drop=True)
    test_labels = labels[test_index]



    train_edges, train_edges_weights, val_graph_pair, edges_val_pn = construct_PosNeg_graphs_v2(train_data, train_labels, val_data, val_labels, topk=3)
    train_edges, train_edges_weights, test_graph_pair, edges_test_pn = construct_PosNeg_graphs_v2(train_data, train_labels, test_data, test_labels, topk=3)




    train_graph = GraphDataset(train_data.values, train_edges, train_labels, edges_weights=train_edges_weights)[0].to(device)
    train_graph = dgl.add_reverse_edges(train_graph)
    # graph = dgl.add_self_loop(graph)

    val_graphs = []

    for idx in range(len(val_index)):
        edges_pos_graph, edges_pos_graph_weights, edges_neg_graph, edges_neg_graph_weights, eval_index_graph = val_graph_pair[idx]
        train_eval_data = np.concatenate((train_data.values, val_data.loc[idx,:].values[np.newaxis,:]), axis=0) 
        train_eval_labels = np.append(train_labels,val_labels[idx])
        val_graph_pos = GraphDataset(train_eval_data, edges_pos_graph, train_eval_labels, test_mask=eval_index_graph,edges_weights=edges_pos_graph_weights)[0].to(device)
        val_graph_pos = dgl.add_reverse_edges(val_graph_pos)
        val_graph_neg = GraphDataset(train_eval_data, edges_neg_graph, train_eval_labels, test_mask=eval_index_graph,edges_weights=edges_neg_graph_weights)[0].to(device)
        val_graph_neg = dgl.add_reverse_edges(val_graph_neg)
        val_graphs.append([val_graph_pos, val_graph_neg])


    val_pn_graphs = []
    for idx in range(len(val_index)):
        edges, edges_weights, eval_index_graph = edges_val_pn[idx]
        train_eval_data = np.concatenate((train_data.values, val_data.loc[idx,:].values[np.newaxis,:]), axis=0) 
        train_eval_labels = np.append(train_labels,val_labels[idx])
        val_pn_graph = GraphDataset(train_eval_data, edges, train_eval_labels, test_mask=eval_index_graph,edges_weights=edges_weights)[0].to(device)
        val_pn_graphs.append(val_pn_graph)
    
    
    test_graphs = []
        
    ''''''
    # 遍历每一个test sample
    # 将其与train_val samples 构建两个图，一个是与pos相连，一个是与neg相连
    # 将构建的图经过网络，预测test sample的属于哪一类，对两个图的结果进行softmax作为其预测结果
    for idx in range(len(test_index)):
        edges_pos_graph, edges_pos_graph_weights, edges_neg_graph, edges_neg_graph_weights, eval_index_graph = test_graph_pair[idx]
        train_eval_data = np.concatenate((train_data.values, test_data.loc[idx,:].values[np.newaxis,:]), axis=0) 
        train_eval_labels = np.append(train_labels,test_labels[idx])
        test_graph_pos = GraphDataset(train_eval_data, edges_pos_graph, train_eval_labels, test_mask=eval_index_graph,edges_weights=edges_pos_graph_weights)[0].to(device)
        test_graph_pos = dgl.add_reverse_edges(test_graph_pos)
        test_graph_neg = GraphDataset(train_eval_data, edges_neg_graph, train_eval_labels, test_mask=eval_index_graph,edges_weights=edges_neg_graph_weights)[0].to(device)
        test_graph_neg = dgl.add_reverse_edges(test_graph_neg)
        test_graphs.append([test_graph_pos, test_graph_neg])

    
    test_pn_graphs = []
    for idx in range(len(test_index)):
        edges, edges_weights, eval_index_graph = edges_test_pn[idx]
        train_eval_data = np.concatenate((train_data.values, test_data.loc[idx,:].values[np.newaxis,:]), axis=0) 
        train_eval_labels = np.append(train_labels,test_labels[idx])
        test_pn_graph = GraphDataset(train_eval_data, edges, train_eval_labels, test_mask=eval_index_graph,edges_weights=edges_weights)[0].to(device)
        test_pn_graphs.append(test_pn_graph)

    
    
    
    
    
    
    
    
    
    
    features = train_graph.ndata['feat']
    


    best_model_wts = copy.deepcopy(model.state_dict())
    lr = hyper_params['learning_rate']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    
    # criterian = nn.CrossEntropyLoss(weight=torch.tensor([1, 6],dtype=torch.float32).to(device))
    criterian = nn.CrossEntropyLoss()

    softmax = nn.Softmax(dim=1)
    # optimizer = torch.optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999],weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5)

    model.train()
    total_loss = 0
    best_val_loss = float("inf")
    best_val_auc = float("-inf")
    best_val_ap = float("-inf")
    
    # writer = SummaryWriter(log_dir='./runs/')
    

    for epoch in range(num_epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # train_sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()

        optimizer.zero_grad()

        outputs = model(train_graph, features, None)

        loss = criterian(outputs, train_graph.ndata['label'])


        probas = softmax(outputs)
        total_loss += loss.item()
        
        

        

        # print('Epoch {} rank {} total loss {}'.format(epoch, rank, total_loss))

        
        loss.backward()
        optimizer.step()
        scheduler.step()

        y_probs = probas.detach().cpu().numpy()[:, 1]
        y_true = train_graph.ndata['label'].detach().cpu().numpy()



        train_auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        train_ap = average_precision_score(y_true=y_true, y_score=y_probs)

        if epoch % 5 == 0:

            val_loss, val_auc, val_ap, _, _ = evaluate_PosNeg_ps_graph_v2(
                model, 
                criterian, 
                val_data,
                val_labels,
                val_graphs)
                    
        
            test_loss, test_auc, test_ap, _, _ = evaluate_PosNeg_ps_graph_v2(
                model, 
                criterian, 
                test_data,
                test_labels,
                test_graphs)


            # val_loss, val_auc, val_ap, _, _ = evaluate_PosNeg_ps_graph_v2(
            #     model, 
            #     criterian, 
            #     val_data,
            #     val_labels,
            #     val_pn_graphs,
            #     pn=True)
                    
        
            # test_loss, test_auc, test_ap, _, _ = evaluate_PosNeg_ps_graph_v2(
            #     model, 
            #     criterian, 
            #     test_data,
            #     test_labels,
            #     test_pn_graphs,
            #     pn=True)
        
            
            # if write_log:
            #     writer.add_scalars('Loss/fold'+str(fold), {'train':total_loss, 'val': val_loss, 'test':test_loss}, epoch)
            #     writer.add_scalars('AUC/fold'+str(fold), {'train':train_auc, 'val': val_auc, 'test':test_auc}, epoch)
            #     writer.add_scalars('AP/fold'+str(fold), {'train':train_ap, 'val': val_ap, 'test':test_ap}, epoch)


            
            print('Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} |'
                    'valid AUC {:.3f}  | valid AP {:.3f} | test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}   {}'.format(epoch, (time.time() - epoch_start_time), total_loss, val_loss, val_auc, val_ap, test_loss, test_auc, test_ap, ('*' if val_auc > best_val_auc else '')))
            print('-' * 89)

            total_loss = 0


            if epoch > 20 and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_wts = copy.deepcopy(model.state_dict())

                path = best_model_dir + 'best_val_model_' + str(fold) + '.' + str(cv_idx) + '.pt'
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                
                torch.save({
                    'model_state_dict': best_model_wts,
                }, path)



    # writer.close()
    return val_auc, val_ap, best_model_wts



def evaluate_PosNeg_ps_graph_v2(
    eval_model,
    criterian,
    eval_data,
    eval_labels,
    test_graphs=None,
    pn=False 
):

    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    
    
    
    eval_model.to(device)

    softmax = nn.Softmax(dim=1)

    eval_model.eval()

    y_probs = np.zeros((len(eval_labels),2))
    y_true = np.zeros((len(eval_labels)))
    total_loss = 0

    if not pn:

        for i, (pos_graph, neg_graph) in enumerate(test_graphs):

            pos_outputs = eval_model(pos_graph, pos_graph.ndata['feat'], None)
            # outputs= outputs.view(-1)
            pos_loss = criterian(pos_outputs[pos_graph.ndata['test_mask']], pos_graph.ndata['label'][pos_graph.ndata['test_mask']])
            # probas = sigmoid(outputs)
            pos_probas = softmax(pos_outputs[pos_graph.ndata['test_mask']])

            neg_outputs = eval_model(neg_graph, neg_graph.ndata['feat'], None)
            # outputs= outputs.view(-1)
            neg_loss = criterian(neg_outputs[neg_graph.ndata['test_mask']], neg_graph.ndata['label'][neg_graph.ndata['test_mask']])
            # probas = sigmoid(outputs)
            neg_probas = softmax(neg_outputs[neg_graph.ndata['test_mask']])


            
            y_probs[i,:] = (pos_probas.detach().cpu().numpy()+ neg_probas.detach().cpu().numpy())/2
            # y_probs[i,:] = pos_probas.detach().cpu().numpy()
            y_true[i] = pos_graph.ndata['label'][pos_graph.ndata['test_mask']].detach().cpu().numpy()
            total_loss += pos_loss.item()

        y_probs = y_probs[:,1]
        auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        ap = average_precision_score(y_true=y_true, y_score=y_probs)
    else:
        for i, graph in enumerate(test_graphs):

            outputs = eval_model(graph, graph.ndata['feat'], graph.edata['edges_weights'])
            # outputs= outputs.view(-1)
            loss = criterian(outputs[graph.ndata['test_mask']], graph.ndata['label'][graph.ndata['test_mask']])
            # probas = sigmoid(outputs)
            probas = softmax(outputs[graph.ndata['test_mask']])

          

            
            y_probs[i,:] = probas.detach().cpu().numpy()
            # y_probs[i,:] = pos_probas.detach().cpu().numpy()
            y_true[i] = graph.ndata['label'][graph.ndata['test_mask']].detach().cpu().numpy()
            total_loss += loss.item()

        y_probs = y_probs[:,1]
        auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        ap = average_precision_score(y_true=y_true, y_score=y_probs)


    return total_loss, auc, ap, y_probs, y_true


def train_PosNeg_ps_graph_v3(
    hyper_params,
    model,  
    data,
    labels,
    train_index,
    val_index,
    train_val_index,
    test_index,
    best_model_dir,
    fold,
    cv_idx,
    write_log=False
):
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model.to(device)

    topk = 3

    train_val_data = data.loc[train_val_index, :].reset_index(inplace=False, drop=True)
    train_val_labels = labels[train_val_index]
    train_data = train_val_data.loc[train_index, :].reset_index(inplace=False, drop=True)
    train_labels = train_val_labels[train_index]
    val_data = train_val_data.loc[val_index, :].reset_index(inplace=False, drop=True)
    val_labels = train_val_labels[val_index]

    test_data = data.loc[test_index, :].reset_index(inplace=False, drop=True)
    test_labels = labels[test_index]



    edges_train, edges_weights_train, edges_val_list, edges_weights_val_list = construct_PosNeg_graphs_v3(train_data, train_labels, val_data, val_labels, topk=topk)
    edges_train, edges_weights_train, edges_test_list, edges_weights_test_list = construct_PosNeg_graphs_v3(train_data, train_labels, test_data, test_labels, topk=topk)




    train_graph = GraphDataset(train_data.values, edges_train, train_labels, edges_weights=edges_weights_train)[0].to(device)
    # train_graph = dgl.add_reverse_edges(train_graph)
    # graph = dgl.add_self_loop(graph)

    val_graphs = []

    for i in range(len(edges_val_list)):
        train_eval_data = np.concatenate((train_data.values, val_data.loc[i,:].values[np.newaxis,:]), axis=0) 
        train_eval_labels = np.append(train_labels,val_labels[i])
        val_graph = GraphDataset(train_eval_data, edges_val_list[i][0], train_eval_labels, test_mask=edges_val_list[i][1],edges_weights=edges_weights_val_list[i])[0].to(device)
        val_graphs.append(val_graph)

    
    test_graphs = []
        
    ''''''
    # 遍历每一个test sample
    # 将其与train_val samples 构建两个图，一个是与pos相连，一个是与neg相连
    # 将构建的图经过网络，预测test sample的属于哪一类，对两个图的结果进行softmax作为其预测结果
    for i in range(len(edges_test_list)):
        train_eval_data = np.concatenate((train_data.values, test_data.loc[i,:].values[np.newaxis,:]), axis=0) 
        train_eval_labels = np.append(train_labels,test_labels[i])
        test_graph = GraphDataset(train_eval_data, edges_test_list[i][0], train_eval_labels, test_mask=edges_test_list[i][1],edges_weights=edges_weights_test_list[i])[0].to(device)
        test_graphs.append(test_graph)

    
    
    
    
    features = train_graph.ndata['feat']
    


    best_model_wts = copy.deepcopy(model.state_dict())
    lr = hyper_params['learning_rate']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    
    # criterian = nn.CrossEntropyLoss(weight=torch.tensor([1, 6],dtype=torch.float32).to(device))
    criterian = nn.CrossEntropyLoss()

    softmax = nn.Softmax(dim=1)
    # optimizer = torch.optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999],weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5)

    model.train()
    total_loss = 0
    best_val_loss = float("inf")
    best_val_auc = float("-inf")
    best_val_ap = float("-inf")
    
    # writer = SummaryWriter(log_dir='./runs/')
    


    fig, ax = plt.subplots()
    for epoch in range(num_epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # train_sampler.set_epoch(epoch)
        
        
        epoch_start_time = time.time()

        optimizer.zero_grad()

        outputs = model(train_graph, features, train_graph.edata['edges_weights'])

        loss = criterian(outputs, train_graph.ndata['label'])


        probas = softmax(outputs)
        total_loss += loss.item()

        train_points = outputs.detach().cpu().numpy()
        

        

        # print('Epoch {} rank {} total loss {}'.format(epoch, rank, total_loss))

        
        loss.backward()
        optimizer.step()
        scheduler.step()

        y_probs = probas.detach().cpu().numpy()[:, 1]
        y_true = train_graph.ndata['label'].detach().cpu().numpy()



        train_auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        train_ap = average_precision_score(y_true=y_true, y_score=y_probs)

        if epoch % 10 == 0:
            

            val_loss, val_auc, val_ap, _, _, val_points = evaluate_PosNeg_ps_graph_v3(
                model, 
                criterian, 
                val_data,
                val_labels,
                val_graphs)
                    
        
            test_loss, test_auc, test_ap, _, _, test_points = evaluate_PosNeg_ps_graph_v3(
                model, 
                criterian, 
                test_data,
                test_labels,
                test_graphs)

            plt.cla()
            ax.scatter(train_points[train_labels==1,0], train_points[train_labels==1,1], color='red')
            ax.scatter(train_points[train_labels==0,0], train_points[train_labels==0,1], color='blue')
            ax.scatter(val_points[val_labels==1,0], val_points[val_labels==1,1], color='orange')
            # ax.scatter(val_points[val_labels==0,0], val_points[val_labels==0,1], color='purple')
            ax.scatter(test_points[test_labels==1,0], test_points[test_labels==1,1], color='green')
            # ax.scatter(test_points[test_labels==0,0], test_points[test_labels==0,1], color='darkblue')

            plt.savefig('./results/pics/train.png')


            # if write_log:
            #     writer.add_scalars('Loss/fold'+str(fold), {'train':total_loss, 'val': val_loss, 'test':test_loss}, epoch)
            #     writer.add_scalars('AUC/fold'+str(fold), {'train':train_auc, 'val': val_auc, 'test':test_auc}, epoch)
            #     writer.add_scalars('AP/fold'+str(fold), {'train':train_ap, 'val': val_ap, 'test':test_ap}, epoch)


            
            print('Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | train AUC {:.3f} | train AP {:.3f} | valid loss {:5.2f} |'
                    'valid AUC {:.3f}  | valid AP {:.3f} | test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}   {}'.format(epoch, (time.time() - epoch_start_time), total_loss, train_auc, train_ap, val_loss, val_auc, val_ap, test_loss, test_auc, test_ap, ('*' if val_auc > best_val_auc else '')))
            print('-' * 89)

            total_loss = 0


            if epoch > 10 and val_auc > best_val_auc:
                best_val_auc = val_auc
            # if epoch > 10 and val_loss < best_val_loss:
            #     best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                path = best_model_dir + 'best_val_model_' + str(fold) + '.' + str(cv_idx) + '.pt'
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                
                torch.save({
                    'model_state_dict': best_model_wts,
                }, path)



    # writer.close()
    return val_auc, val_ap, best_model_wts



def evaluate_PosNeg_ps_graph_v3(
    eval_model,
    criterian,
    eval_data,
    eval_labels,
    test_graphs=None
):

    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    
    
    
    eval_model.to(device)

    softmax = nn.Softmax(dim=1)

    eval_model.eval()

    y_probs = np.zeros((len(eval_labels),2))
    y_true = np.zeros((len(eval_labels)))
    eval_points = np.zeros((len(eval_labels),2))
    total_loss = 0

    for i, graph in enumerate(test_graphs):

        outputs = eval_model(graph, graph.ndata['feat'], graph.edata['edges_weights'])
        # outputs= outputs.view(-1)
        loss = criterian(outputs[graph.ndata['test_mask']], graph.ndata['label'][graph.ndata['test_mask']])
        # probas = sigmoid(outputs)
        probas = softmax(outputs[graph.ndata['test_mask']])

        

        
        y_probs[i,:] = probas.detach().cpu().numpy()
        eval_points[i,:] = outputs[graph.ndata['test_mask']].detach().cpu().numpy()
        # y_probs[i,:] = pos_probas.detach().cpu().numpy()
        y_true[i] = graph.ndata['label'][graph.ndata['test_mask']].detach().cpu().numpy()
        total_loss += loss.item()
    total_loss /= len(test_graphs)

    y_probs = y_probs[:,1]
    auc = roc_auc_score(y_true=y_true, y_score=y_probs)
    ap = average_precision_score(y_true=y_true, y_score=y_probs)



    


    return total_loss, auc, ap, y_probs, y_true, eval_points


def train_PosNeg_ps_graph_v4(
    hyper_params,
    model,  
    data,
    labels,
    train_index,
    val_index,
    train_val_index,
    test_index,
    best_model_dir,
    fold,
    cv_idx,
    write_log=False
):
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model.to(device)

    topk = 3

    train_val_data = data.loc[train_val_index, :].reset_index(inplace=False, drop=True)
    train_val_labels = labels[train_val_index]
    train_data = train_val_data.loc[train_index, :].reset_index(inplace=False, drop=True)
    train_labels = train_val_labels[train_index]
    val_data = train_val_data.loc[val_index, :].reset_index(inplace=False, drop=True)
    val_labels = train_val_labels[val_index]

    test_data = data.loc[test_index, :].reset_index(inplace=False, drop=True)
    test_labels = labels[test_index]



    edges_train, edges_weights_train, edges_val_list, edges_weights_val_list = construct_PosNeg_graphs_v4(train_data, train_labels, val_data, val_labels, topk=topk)
    edges_train, edges_weights_train, edges_test_list, edges_weights_test_list = construct_PosNeg_graphs_v4(train_data, train_labels, test_data, test_labels, topk=topk)




    train_graph = GraphDataset(train_data.values, edges_train, train_labels, edges_weights=edges_weights_train)[0].to(device)
    # train_graph = dgl.add_reverse_edges(train_graph)
    # graph = dgl.add_self_loop(graph)

    val_graphs = []

    for i in range(len(edges_val_list)):
        train_eval_data = np.concatenate((train_data.values, val_data.loc[i,:].values[np.newaxis,:]), axis=0) 
        train_eval_labels = np.append(train_labels,val_labels[i])
        val_graph = GraphDataset(train_eval_data, edges_val_list[i][0], train_eval_labels, test_mask=edges_val_list[i][1],edges_weights=edges_weights_val_list[i])[0].to(device)
        val_graphs.append(val_graph)

    
    test_graphs = []
        
    ''''''
    # 遍历每一个test sample
    # 将其与train_val samples 构建两个图，一个是与pos相连，一个是与neg相连
    # 将构建的图经过网络，预测test sample的属于哪一类，对两个图的结果进行softmax作为其预测结果
    for i in range(len(edges_test_list)):
        train_eval_data = np.concatenate((train_data.values, test_data.loc[i,:].values[np.newaxis,:]), axis=0) 
        train_eval_labels = np.append(train_labels,test_labels[i])
        test_graph = GraphDataset(train_eval_data, edges_test_list[i][0], train_eval_labels, test_mask=edges_test_list[i][1],edges_weights=edges_weights_test_list[i])[0].to(device)
        test_graphs.append(test_graph)

    
    
    
    
    features = train_graph.ndata['feat']
    


    best_model_wts = copy.deepcopy(model.state_dict())
    lr = hyper_params['learning_rate']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    
    # criterian = nn.CrossEntropyLoss(weight=torch.tensor([1, 6],dtype=torch.float32).to(device))
    criterian = nn.CrossEntropyLoss()

    softmax = nn.Softmax(dim=1)
    # optimizer = torch.optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999],weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5)

    model.train()
    total_loss = 0
    best_val_loss = float("inf")
    best_val_auc = float("-inf")
    best_val_ap = float("-inf")
    
    # writer = SummaryWriter(log_dir='./runs/')
    


    fig, ax = plt.subplots()
    for epoch in range(num_epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # train_sampler.set_epoch(epoch)
        
        
        epoch_start_time = time.time()

        optimizer.zero_grad()

        outputs = model(train_graph, features, train_graph.edata['edges_weights'])

        loss = criterian(outputs, train_graph.ndata['label'])


        probas = softmax(outputs)
        total_loss += loss.item()

        train_points = outputs.detach().cpu().numpy()
        

        

        # print('Epoch {} rank {} total loss {}'.format(epoch, rank, total_loss))

        
        loss.backward()
        optimizer.step()
        scheduler.step()

        y_probs = probas.detach().cpu().numpy()[:, 1]
        y_true = train_graph.ndata['label'].detach().cpu().numpy()



        train_auc = roc_auc_score(y_true=y_true, y_score=y_probs)
        train_ap = average_precision_score(y_true=y_true, y_score=y_probs)

        if epoch % 100 == 0:
            

            val_loss, val_auc, val_ap, _, _, val_points = evaluate_PosNeg_ps_graph_v4(
                model, 
                criterian, 
                val_data,
                val_labels,
                val_graphs)
                    
        
            test_loss, test_auc, test_ap, _, _, test_points = evaluate_PosNeg_ps_graph_v4(
                model, 
                criterian, 
                test_data,
                test_labels,
                test_graphs)

            plt.cla()
            ax.scatter(train_points[train_labels==1,0], train_points[train_labels==1,1], color='red')
            ax.scatter(train_points[train_labels==0,0], train_points[train_labels==0,1], color='blue')
            # ax.scatter(val_points[val_labels==1,0], val_points[val_labels==1,1], color='orange')
            # ax.scatter(val_points[val_labels==0,0], val_points[val_labels==0,1], color='purple')
            ax.scatter(test_points[test_labels==1,0], test_points[test_labels==1,1], color='green')
            ax.scatter(test_points[test_labels==0,0], test_points[test_labels==0,1], color='darkblue')

            plt.savefig('./results/pics/train.png')


            # if write_log:
            #     writer.add_scalars('Loss/fold'+str(fold), {'train':total_loss, 'val': val_loss, 'test':test_loss}, epoch)
            #     writer.add_scalars('AUC/fold'+str(fold), {'train':train_auc, 'val': val_auc, 'test':test_auc}, epoch)
            #     writer.add_scalars('AP/fold'+str(fold), {'train':train_ap, 'val': val_ap, 'test':test_ap}, epoch)


            
            print('Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | train AUC {:.3f} | train AP {:.3f} | valid loss {:5.2f} |'
                    'valid AUC {:.3f}  | valid AP {:.3f} | test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}   {}'.format(epoch, (time.time() - epoch_start_time), total_loss, train_auc, train_ap, val_loss, val_auc, val_ap, test_loss, test_auc, test_ap, ('*' if val_auc > best_val_auc else '')))
            print('-' * 89)

            total_loss = 0


            if epoch > 10 and val_auc > best_val_auc:
                best_val_auc = val_auc
            # if epoch > 10 and val_loss < best_val_loss:
            #     best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                path = best_model_dir + 'best_val_model_' + str(fold) + '.' + str(cv_idx) + '.pt'
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                
                torch.save({
                    'model_state_dict': best_model_wts,
                }, path)



    # writer.close()
    return val_auc, val_ap, best_model_wts



def evaluate_PosNeg_ps_graph_v4(
    eval_model,
    criterian,
    eval_data,
    eval_labels,
    test_graphs=None
):

    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    
    
    
    eval_model.to(device)

    softmax = nn.Softmax(dim=1)

    eval_model.eval()

    y_probs = np.zeros((len(eval_labels),2))
    y_true = np.zeros((len(eval_labels)))
    eval_points = np.zeros((len(eval_labels),2))
    total_loss = 0

    for i, graph in enumerate(test_graphs):

        outputs = eval_model(graph, graph.ndata['feat'], graph.edata['edges_weights'])
        # outputs= outputs.view(-1)
        loss = criterian(outputs[graph.ndata['test_mask']], graph.ndata['label'][graph.ndata['test_mask']])
        # probas = sigmoid(outputs)
        probas = softmax(outputs[graph.ndata['test_mask']])

        

        
        y_probs[i,:] = probas.detach().cpu().numpy()
        eval_points[i,:] = outputs[graph.ndata['test_mask']].detach().cpu().numpy()
        # y_probs[i,:] = pos_probas.detach().cpu().numpy()
        y_true[i] = graph.ndata['label'][graph.ndata['test_mask']].detach().cpu().numpy()
        total_loss += loss.item()
    total_loss /= len(test_graphs)

    y_probs = y_probs[:,1]
    auc = roc_auc_score(y_true=y_true, y_score=y_probs)
    ap = average_precision_score(y_true=y_true, y_score=y_probs)



    


    return total_loss, auc, ap, y_probs, y_true, eval_points