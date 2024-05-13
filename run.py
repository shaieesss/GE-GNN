import argparse
import os
import sys
import dgl
import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
import torch.optim as optim
import yaml
import random
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score
import copy
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tfinance')
    args = parser.parse_args()
    config_path = './config/'+args.dataset+'.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace(**config)
    print('----------------------------------')
    print('              args')
    print('----------------------------------')
    print(f'dataset:\t{args.dataset}')
    print(f'seed:\t{args.seed}')
    print(f'epoch:\t{args.epoch}')
    print(f'early_stop:\t{args.early_stop}')
    print(f'lr:\t{args.lr}')
    print(f'weigth_decay:{args.weight_decay}')
    print(f'gamma1:\t{args.gamma1}')
    print(f'gamma2:\t{args.gamma2}')
    print(f'intra_dim:\t{args.intra_dim}')
    print(f'head:\t{args.head}')
    print(f'n_layer:\t{args.n_layer}')
    print(f'dropout:\t{args.dropout}')
    print(f'cuda:\t{args.cuda}')
    print('----------------------------------')
    return args
args = parse_arg()
class EarlyStop():
    def __init__(self, early_stop, if_more=True) -> None:
        self.best_eval = 0
        self.best_epoch = 0
        self.if_more = if_more
        self.early_stop = early_stop
        self.stop_steps = 0
    
    def step(self, current_eval, current_epoch):
        do_stop = False
        do_store = False
        if self.if_more:
            if current_eval > self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        else:
            if current_eval < self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        return do_store, do_stop

def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5
def prob2pred(prob, threshhold=0.5):
    pred = np.zeros_like(prob, dtype=np.int32)
    pred[prob >= threshhold] = 1
    pred[prob < threshhold] = 0
    return pred
def evaluate(labels, logits, result_path = ''):
    probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    if len(result_path)>0:
        np.save(result_path+'_result_preds', preds)
        np.save(result_path+'_result_probs'
                , probs)
    conf = confusion_matrix(labels, preds)
    recall = recall_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    auc = roc_auc_score(labels, probs)
    gmean = conf_gmean(conf)
    return f1_macro, auc, gmean, recall

def hinge_loss(labels, scores):
    margin = 1
    ls = labels*scores
    
    loss = F.relu(margin-ls)
    loss = loss.mean()
    return loss

def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx
class H_layer(nn.Module):
    def __init__(self, input_dim, output_dim, head, relation_aware, etype, dropout, if_sum=False):
        super().__init__()
        self.etype = etype
        self.head = head
        self.hd = output_dim
        self.if_sum = if_sum
        self.atten = nn.Linear(3*self.hd, 1)
        self.relu = nn.ReLU()
        self.relation_ware = relation_aware
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.w_liner = nn.Linear(input_dim, output_dim*head)

        
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['feat'] = h
            g.apply_edges(self.sign_edges, etype=self.etype)
            edge_sum = g.edata['edge_sum']
            h = self.w_liner(h)
            g.ndata['h'] = h
            g.update_all(message_func=self.message, reduce_func=self.reduce, etype=self.etype)
            out = g.ndata['out']
            edge_s = g.ndata['s']
            if not self.if_sum:
                return edge_s, out, h.view(-1, self.head*self.hd)
            else:
                return edge_s, out, h.view(-1, self.head, self.hd).sum(-2)

        
    def message(self, edges):
        src_f = edges.src['h']
        src_f = src_f.view(-1, self.head, self.hd)
        dst_f = edges.dst['h']
        dst_f = dst_f.view(-1, self.head, self.hd)
        edge_s = edges.data['edge_sum'].view(-1, self.head, self.hd)
        z = torch.cat([src_f, dst_f, edge_s], dim=-1)
        
        alpha = self.atten(z)
        alpha = self.leakyrelu(alpha)
        return {'atten':alpha, 'sf':src_f, 'edge_s': edge_s}


    def reduce(self, nodes):
        alpha = nodes.mailbox['atten']
        sf = nodes.mailbox['sf']
        alpha = self.softmax(alpha)
        out = torch.sum(alpha*sf, dim=1)
        if not self.if_sum:
            out = out.view(-1, self.head*self.hd)
            edge_s = torch.mean(nodes.mailbox['edge_s'], dim=1).view(-1, self.head*self.hd)
            return {'out':out, 's': edge_s}
        else:
            out = out.sum(dim=-2)
            edge_s = torch.sum(torch.mean(nodes.mailbox['edge_s'], dim=1), dim=-2)

            return {'out':out, 's': edge_s}


    def sign_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        edge_sum = self.relation_ware(src, dst)
        return {'edge_sum':edge_sum} 
    
class Gate(nn.Module):
    def __init__(self, head, output_dim, dropout, if_sum=False):
        super().__init__()
        self.output_dim = output_dim
        self.head = head
        if not if_sum:
            self.beta = nn.Parameter(torch.empty(size=(2*self.head*self.output_dim, 1)))
            nn.init.xavier_normal_(self.beta.data, gain=1.414)
        else:
            
            self.beta = nn.Parameter(torch.empty(size=(2*self.output_dim, 1))) 
            nn.init.xavier_normal_(self.beta.data, gain=1.414)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, edge_sum, out, h):
        beta = torch.cat([edge_sum, out], dim=1)
        gate = self.sigmoid(torch.matmul(beta, self.beta))
        final = gate * out + (1 - gate) * h
        return final
    

class RelationAware(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.d_liner = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src-dst
        edge_sum = self.tanh(src + dst + diff)
        return edge_sum


class MultiRelationGE_GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, head, dataset, dropout, if_sum=False):
        super().__init__()
        self.relation = copy.deepcopy(dataset.etypes)
        if 'homo' in self.relation:
            self.relation.remove('homo')
        self.n_relation = len(self.relation)
        self.if_sum = if_sum
        if not self.if_sum:
            self.liner = nn.Linear(self.n_relation*output_dim*head, output_dim*head)
        else:
            self.liner = nn.Linear(self.n_relation*output_dim, output_dim)
        self.relation_aware = RelationAware(input_dim, output_dim*head, dropout)
        
        self.minelayers = nn.ModuleDict()
        self.sublayer = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        
#         if not self.if_sum:
        for e in self.relation:
            self.sublayer = nn.ModuleList()
            self.minelayers[e] = self.sublayer
            self.sublayer.append(H_layer(input_dim, output_dim, head, self.relation_aware, e, dropout, if_sum))
            self.sublayer.append(Gate(head, output_dim, dropout, if_sum))
    
    def forward(self, g, h):
        hs = []
        for e in self.relation:
            edge_sum1, out1, h1 = self.minelayers[e][0](g, h)
            he = self.minelayers[e][1](edge_sum1, out1, h1)
            hs.append(he)
        x = torch.cat(hs, dim=1)
        x = self.dropout(x)
        x = self.liner(x)
        return x

    def loss(self, g, h):
        with g.local_scope():
            g.ndata['feat'] = h
            train_mask = g.ndata['train_mask'].bool()
#             train_h = agg_h[train_mask]
            train_label = g.ndata['label'][train_mask]
            train_pos = train_label==1
            train_neg = train_label==0
            train_pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = np.random.choice(train_neg_index, size=len(train_pos_index))
            node_index = np.concatenate([train_neg_index, train_pos_index])
            node_index.sort()
            
            hs = []
            diff_loss = 0
            for e in self.relation:
                edge_sum1, out1, h1 = self.minelayers[e][0](g, h)
                he = self.minelayers[e][1](edge_sum1, out1, h1)
                hs.append(he)
                
#                 diff_loss += diff_loss1
            x = torch.cat(hs, dim=1)
            x = self.dropout(x)
            agg_h = self.liner(x)
#             diff_loss1 = F.cross_entropy(agg_h[train_mask][node_index], train_label[node_index])
            if not self.if_sum:
                diff_loss1 = F.cross_entropy(agg_h[train_mask][node_index], train_label[node_index])
            else:
                diff_loss1 = F.cross_entropy(agg_h[train_mask][node_index], train_label[node_index])
#             diff_loss2 = F.cross_entropy(self.filter2(h[train_mask][node_index]), train_label[node_index])
#             diff_loss3 = F.cross_entropy(h[train_mask][node_index], train_label[node_index])
            return agg_h, diff_loss1
            


class GE_GNN(nn.Module):
    def __init__(self, args, g):
        super().__init__()
        self.n_layer = args.n_layer
        self.input_dim = g.nodes['r'].data['feature'].shape[1]
        self.intra_dim = args.intra_dim
        self.n_class = args.n_class
        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2
        self.n_layer = args.n_layer
        self.mine_layers = nn.ModuleList()
        if args.n_layer == 1:
            self.mine_layers.append(MultiRelationGE_GNNLayer(self.input_dim, self.n_class, args.head, g, args.dropout, if_sum=True))
        else:
            self.mine_layers.append(MultiRelationGE_GNNLayer(self.input_dim, self.intra_dim, args.head, g, args.dropout))
            for _ in range(1, self.n_layer-1):
                self.mine_layers.append(MultiRelationGE_GNNLayer(self.intra_dim*args.head, self.intra_dim, args.head, g, args.dropout))
            self.mine_layers.append(MultiRelationGE_GNNLayer(self.intra_dim*args.head, self.n_class, args.head, g, args.dropout, if_sum=True))
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, g):
        feats = g.ndata['feature'].float()
        h = self.mine_layers[0](g, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h = self.mine_layers[i](g, h)
                h = self.relu(h)
                h = self.dropout(h)

            h = self.mine_layers[-1](g, h)
        return h 
    
    def loss(self, g):  
        feats = g.ndata['feature'].float()
        train_mask = g.ndata['train_mask'].bool()
        train_label = g.ndata['label'][train_mask]
        train_pos = train_label == 1
        train_neg = train_label == 0
        
        pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
        neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
        neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        index = np.concatenate([pos_index, neg_index])
        index.sort()
        h, prototype_loss = self.mine_layers[0].loss(g, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h, p_loss = self.mine_layers[i].loss(g, h)
                h = self.relu(h)
                h = self.dropout(h)
                prototype_loss += p_loss
            h, p_loss = self.mine_layers[-1].loss(g, h)
            prototype_loss += p_loss
        model_loss = F.cross_entropy(h[train_mask][index], train_label[index])
        loss = model_loss + 1.2 * prototype_loss / 3
        return loss
setup_seed(args.seed)
device = torch.device(args.cuda)
args.device = device
dataset_path = args.data_path+args.dataset+'.dgl'
model_path = args.result_path+args.dataset+'_model_head8.pt'
results = {'F1-macro':[],'AUC':[],'G-Mean':[],'recall':[]}
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

# load dataset and normalize feature

dataset = dgl.load_graphs(dataset_path)[0][0]
features = dataset.ndata['feature'].numpy()
features = normalize(features)
dataset.ndata['feature'] = torch.from_numpy(features).float()
dataset = dataset.to(device)

'''
# train model
'''
print('Start training model...')
model = GE_GNN(args, dataset)
model = model.to(device)
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
early_stop = EarlyStop(args.early_stop)
for e in range(args.epoch):

    model.train()
    loss = model.loss(dataset)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        '''
        # valid
        '''
        model.eval()
        valid_mask = dataset.ndata['valid_mask'].bool()
        valid_labels = dataset.ndata['label'][valid_mask].cpu().numpy()
        valid_logits = model(dataset)[valid_mask]
        valid_preds = valid_logits.argmax(1).cpu().numpy()

        f1_macro, auc, gmean, recall = evaluate(valid_labels, valid_logits)

        if args.log:
            print(f'{e}: Best Epoch:{early_stop.best_epoch}, Best valid AUC:{early_stop.best_eval}, Loss:{loss.item()}, Current valid: Recall:{recall}, F1_macro:{f1_macro}, G-Mean:{gmean}, AUC:{auc}')
        do_store, do_stop = early_stop.step(auc, e)
        if do_store:
            torch.save(model, model_path)
        if do_stop:
            break
print('End training')
'''
# test model
'''
print('Test model...')
model = torch.load(model_path)      
with torch.no_grad():
    model.eval()
    test_mask = dataset.ndata['test_mask'].bool()
    test_labels = dataset.ndata['label'][test_mask]
    test_labels = test_labels.cpu().numpy()
    logits = model(dataset)[test_mask]
    logits = logits.cpu()
    test_result_path = args.result_path+args.dataset
    f1_macro, auc, gmean, recall = evaluate(test_labels, logits, test_result_path)
    results['F1-macro'].append(f1_macro)
    results['AUC'].append(auc)
    results['G-Mean'].append(gmean)
    results['recall'].append(recall)
    print(f'Test: F1-macro:{f1_macro}, AUC:{auc}, G-Mean:{gmean}, Recall:{recall}')
# exit()

