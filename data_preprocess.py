import argparse
import os
import sys
import dgl
import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
import pickle
import random
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, average_precision_score, precision_score
from collections import defaultdict
# from catboost import CatBoostClassifier
import networkx as nx
from dgl.data.utils import load_graphs, save_graphs
import os

import multiprocessing
import random
import toad
from sklearn.metrics._ranking import _binary_clf_curve


def index_to_mask(index_list, length):
    """
    将给定的索引列表转换为一个长度为length的掩码张量。

    参数:
        index_list (list): 包含要转换为掩码的索引的列表。
        length (int): 掩码张量的长度。

    返回:
        mask (torch.Tensor): 一个长度为length的布尔掩码张量，其中给定索引的位置为True，其他位置为False。
    """
    mask = torch.zeros(length, dtype=torch.bool)
    mask[index_list] = True
    return mask


def mask_to_index(mask):
    """
    将给定的掩码张量转换为一个索引列表。

    参数:
        mask (torch.Tensor): 一个布尔掩码张量。

    返回:
        index_list (list): 包含掩码中True值对应的索引的列表。
    """
    index_list = torch.nonzero(mask).squeeze()
    return index_list


# 设置随机种子
def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def describe(graph):
    # 统计输出
    num_nodes = graph.ndata['feat'].shape[0]
    num_features = graph.ndata['feat'].shape[1]
    # 计算正类（欺诈样本）和负类（非欺诈样本）的数量
    num_positive = torch.sum(graph.ndata['label'] == 1)
    num_negative = torch.sum(graph.ndata['label'] == 0)
    # 计算欺诈样本占比
    fraud_ratio = num_positive / (num_positive + num_negative)
    print(f"总样本数: {num_nodes}")
    print(f"欺诈样本数: {num_positive}")
    print(f"非欺诈样本数: {num_negative}")
    print(f"欺诈样本占比: {fraud_ratio:.2%}")
    print(f"特征个数: {num_features}")
    print()
    for etype in graph.etypes:
        subgraph = graph.edge_type_subgraph([etype])
        num_edges = subgraph.number_of_edges()
        avg_out_degree = np.mean(subgraph.out_degrees().numpy())

        # 输出统计信息
        print(f"{etype}边关系下的统计信息:")
        print(f"边的个数: {num_edges}")
        print(f"平均出度: {avg_out_degree:.2f}")

    print("\n数据划分情况:")
    print(
        f"训练集: {graph.ndata['trn_msk'].sum()} 验证集: {graph.ndata['val_msk'].sum()} 测试集: {graph.ndata['tst_msk'].sum()}")
    print(
        f"训练集: {graph.ndata['trn_msk'].sum() / num_nodes:.2%} 验证集: {graph.ndata['val_msk'].sum() / num_nodes:.2%} 测试集: {graph.ndata['tst_msk'].sum() / num_nodes:.2%}")


# 计算获得最优的macrof1,gmean和对应的阈值
def get_max_macrof1_gmean(true, prob):
    fps, tps, thresholds = _binary_clf_curve(true, prob)
    n_pos = np.sum(true)
    n_neg = len(true) - n_pos
    fns = n_pos - tps
    tns = n_neg - fps

    f11 = 2 * tps / (2 * tps + fns + fps)
    f10 = 2 * tns / (2 * tns + fns + fps)
    marco_f1 = (f11 + f10) / 2

    idx = np.argmax(marco_f1)
    best_marco_f1 = marco_f1[idx]
    best_marco_f1_thr = thresholds[idx]

    gmean = np.sqrt(tps / n_pos * tns / n_neg)
    idx = np.argmax(gmean)
    best_gmean = gmean[idx]
    best_gmean_thr = thresholds[idx]
    return best_marco_f1, best_marco_f1_thr, best_gmean, best_gmean_thr


# 计算所有metrics指标
def cal_metrics(prob, y, trn_idx, val_idx, tst_idx, verbose=False):
    out_dic = {}
    val_th1 = 0
    val_th2 = 0
    for prefix, idx in zip(['final_trn/', 'final_val/', 'final_tst/'], [trn_idx, val_idx, tst_idx]):
        prob_ = prob[idx]
        y_ = y[idx]

        if prefix in ['final_trn/', 'final_val/']:
            mf1, th1, gme, th2 = get_max_macrof1_gmean(y_, prob_)
            val_th1 = th1
            val_th2 = th2
            pred = np.where(prob_ > th1, 1, 0)
        elif 'tst' in prefix:
            th1 = val_th1
            th2 = val_th2
            pred = np.where(prob_ > th1, 1, 0)
            mf1 = f1_score(y_true=y_, y_pred=pred, average='macro')
            tn, fp, fn, tp = confusion_matrix(y_, pred).ravel()
            gme = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

        rec = recall_score(y_, pred)
        pre = precision_score(y_, pred)
        auc = roc_auc_score(y_, prob_)
        aps = average_precision_score(y_, prob_)

        dic = {
            f'{prefix}auc': np.round(auc, 5),
            f'{prefix}aps': np.round(aps, 5),  # AP score
            f'{prefix}mf1': np.round(mf1, 5),
            f'{prefix}th1': np.round(th1, 5),
            f'{prefix}gme': np.round(gme, 5),
            f'{prefix}th2': np.round(th2, 5),
            f'{prefix}rec': np.round(rec, 5),
            f'{prefix}pre': np.round(pre, 5),
        }
        formatted_dic = {k: f"{v:.5f}" for k, v in dic.items()}
        if verbose == True:
            print(formatted_dic)
        out_dic.update(dic)
    return out_dic


# 决策树分箱编码
def bin_encoding2(graph, trn_idx, n_bins, BCD=False, col_index=None):
    X = graph.ndata['feat'].numpy()
    y = graph.ndata['label'].numpy()
    X = pd.DataFrame(X)
    trn_X = X.iloc[trn_idx]
    trn_y = pd.DataFrame(y[trn_idx])
    combiner = toad.transform.Combiner()
    combiner.fit(trn_X, trn_y, method='dt', min_samples=0.01, n_bins=n_bins, )
    bins = combiner.export()
    if col_index is None or col_index == 'None':
        col_index = X.columns
    bin_encoded_X = combiner.transform(X[col_index])

    bin_encoded_X_dummies = pd.get_dummies(bin_encoded_X, columns=col_index)
    feature = pd.concat([X, bin_encoded_X_dummies], axis=1)

    feature = feature.astype(float)
    return feature
def generate_edges_labels(edges, labels, train_idx):
    row, col = edges
    edge_labels = []
    edge_train_mask = []
    
    for i, j in zip(row, col):
        i = i.item()
        j = j.item()
        if labels[i] == labels[j]:
            edge_labels.append(1)
        else:
            edge_labels.append(-1)
        if i in train_idx and j in train_idx:
            edge_train_mask.append(1)
        else:
            edge_train_mask.append(0)
    edge_labels = torch.Tensor(edge_labels).long()
    edge_train_mask = torch.Tensor(edge_train_mask).bool()
    return edge_labels, edge_train_mask

if __name__ == '__main__':
    original_path = './originaldata/'
    dataset_path = './data/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    args = parser.parse_args()
    print('**********************************')
    print(f'Generate {args.dataset}')
    print('**********************************')
    if args.dataset == 'yelp':
        '''
        # generate yelp dataset
        '''
        if os.path.exists(dataset_path+'yelp.dgl'):
            print('Dataset yelp has been created')
            sys.exit()
        print('Convert to DGL Graph.')
        yelp_path = dataset_path+'YelpChi.mat'
        yelp = scio.loadmat(yelp_path)
        feats = yelp['features'].todense()
        features = torch.from_numpy(feats)
        lbs = yelp['label'][0]
        labels = torch.from_numpy(lbs)
        homo = yelp['homo']
        homo = homo+homo.transpose()
        homo = homo.tocoo()
        rur = yelp['net_rur']
        rur = rur+rur.transpose()
        rur = rur.tocoo()
        rtr = yelp['net_rtr']
        rtr = rtr+rtr.transpose()
        rtr = rtr.tocoo()
        rsr = yelp['net_rsr']
        rsr = rsr+rsr.transpose()
        rsr = rsr.tocoo()
        
        yelp_graph_structure = {
            ('r','homo','r'):(torch.tensor(homo.row), torch.tensor(homo.col)),
            ('r','u','r'):(torch.tensor(rur.row), torch.tensor(rur.col)),
            ('r','t','r'):(torch.tensor(rtr.row), torch.tensor(rtr.col)),
            ('r','s','r'):(torch.tensor(rsr.row), torch.tensor(rsr.col))
        }
        yelp_graph = dgl.heterograph(yelp_graph_structure)
        for t in yelp_graph.etypes:
            yelp_graph.remove_self_loop(etype=t)
            yelp_graph.add_self_loop(etype=t)
        yelp_graph.nodes['r'].data['feature'] = features
        yelp_graph.nodes['r'].data['label'] = labels
        print('Generate dataset partition.')
        train_ratio = 0.4
        test_ratio = 0.67
        index = list(range(len(lbs)))
        dataset_l = len(lbs)
        train_idx, rest_idx, train_lbs, rest_lbs = train_test_split(index, lbs, stratify=lbs, train_size=train_ratio, random_state=2, shuffle=True)
        valid_idx, test_idx, _,_ = train_test_split(rest_idx, rest_lbs, stratify=rest_lbs, test_size=test_ratio, random_state=2, shuffle=True)
        train_mask = torch.zeros(dataset_l, dtype=torch.bool)
        train_mask[np.array(train_idx)] = True
        valid_mask = torch.zeros(dataset_l, dtype=torch.bool)
        valid_mask[np.array(valid_idx)] = True
        test_mask = torch.zeros(dataset_l, dtype=torch.bool)
        test_mask[np.array(test_idx)] = True
        
        yelp_graph.nodes['r'].data['train_mask'] = train_mask
        yelp_graph.nodes['r'].data['valid_mask'] = valid_mask
        yelp_graph.nodes['r'].data['test_mask'] = test_mask
        
        print('Generate edge labels.')
        homo_edges = yelp_graph.edges(etype='homo')
        homo_labels, homo_train_mask = generate_edges_labels(homo_edges, lbs, train_idx)
        yelp_graph.edges['homo'].data['label'] = homo_labels
        yelp_graph.edges['homo'].data['train_mask'] = homo_train_mask
        
        dgl.save_graphs(dataset_path+'yelp.dgl', yelp_graph)
        print(f'yelp dataset\'s num nodes:{yelp_graph.num_nodes("r")}, \
            rur edges:{yelp_graph.num_edges("u")}, \
            rtr edges:{yelp_graph.num_edges("t")}, \
            rsr edges:{yelp_graph.num_edges("s")}')
        print(f'Edge train num:{homo_train_mask.sum().item()}, pos num:{(homo_labels[homo_train_mask]==1).sum().item()}')
        
    elif args.dataset == 'amazon':
        '''
        # generate amazon dataset
        '''
        if os.path.exists(dataset_path+'amazon.dgl'):
            print('dataset amazon has been created')
            sys.exit()
        print('Convert to DGL Graph.')
        amazon_path = dataset_path+'Amazon.mat'
        amazon = scio.loadmat(amazon_path)
        feats = amazon['features'].todense()
        features = torch.from_numpy(feats).float()
        lbs = amazon['label'][0]
        labels = torch.from_numpy(lbs).long()
        homo = amazon['homo']
        homo = homo+homo.transpose()
        homo = homo.tocoo()
        upu = amazon['net_upu']
        upu = upu+upu.transpose()
        upu = upu.tocoo()
        usu = amazon['net_usu']
        usu = usu+usu.transpose()
        usu = usu.tocoo()
        uvu = amazon['net_uvu']
        uvu = uvu+uvu.transpose()
        uvu = uvu.tocoo()
        
        amazon_graph_structure = {
            ('r','homo','r'):(torch.tensor(homo.row), torch.tensor(homo.col)),
            ('r','p','r'):(torch.tensor(upu.row), torch.tensor(upu.col)),
            ('r','s','r'):(torch.tensor(usu.row), torch.tensor(usu.col)),
            ('r','v','r'):(torch.tensor(uvu.row), torch.tensor(uvu.col))
        }
        amazon_graph = dgl.heterograph(amazon_graph_structure)
        for t in amazon_graph.etypes:
            amazon_graph.remove_self_loop(t)
            amazon_graph.add_self_loop(t)
        amazon_graph.nodes['r'].data['feature'] = features
        amazon_graph.nodes['r'].data['label'] = labels
        print('Generate dataset partition.')
        train_ratio = 0.4
        test_ratio = 0.67
        index = list(range(3305, len(labels)))
        dataset_l = len(lbs)
        train_idx, rest_idx, train_lbs, rest_lbs = train_test_split(index, lbs[3305:], stratify=lbs[3305:], train_size=train_ratio, random_state=2, shuffle=True)
        valid_idx, test_idx, _,_ = train_test_split(rest_idx, rest_lbs, stratify=rest_lbs, test_size=test_ratio, random_state=2, shuffle=True)
        train_mask = torch.zeros(dataset_l, dtype=torch.bool)
        train_mask[np.array(train_idx)] = True
        valid_mask = torch.zeros(dataset_l, dtype=torch.bool)
        valid_mask[np.array(valid_idx)] = True
        test_mask = torch.zeros(dataset_l, dtype=torch.bool)
        test_mask[np.array(test_idx)] = True
        
        amazon_graph.nodes['r'].data['train_mask'] = train_mask
        amazon_graph.nodes['r'].data['valid_mask'] = valid_mask
        amazon_graph.nodes['r'].data['test_mask'] = test_mask
        
        print('Generate edge labels.')
        homo_edges = amazon_graph.edges(etype='homo')
        homo_labels, homo_train_mask = generate_edges_labels(homo_edges, lbs, train_idx)
        amazon_graph.edges['homo'].data['label'] = homo_labels
        amazon_graph.edges['homo'].data['train_mask'] = homo_train_mask
        
        dgl.save_graphs(dataset_path+'amazon.dgl', amazon_graph)
        print(f'amazon dataset\'s num nodes:{amazon_graph.num_nodes("r")}, \
            upu edges:{amazon_graph.num_edges("p")}, \
            usu edges:{amazon_graph.num_edges("s")}, \
            uvu edges:{amazon_graph.num_edges("v")}')
        print(f'Edge train num:{homo_train_mask.sum().item()}, pos num:{(homo_labels[homo_train_mask]==1).sum().item()}')
    print('***************endl****************')


