import os
import sys
import numpy as np
import torch
import pickle as pkl
import scipy.sparse as sp
import dgl

def build_graph(type, node):
    g = dgl.DGLGraph()
    # add 34 nodes in to the data; nodes are labeled from 0~33
    g.add_nodes(node)
    edge_list = []
    if type == 'direct':
        with open('./data/ASSIST09/82/graph/K_Directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        # g.add_edges(dst, src)
        return g
    elif type == 'undirect':
        with open('./data/ASSIST09/82/graph/K_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        g.add_edges(dst, src)
        return g
    elif type == 'k_from_e':
        with open('./data/ASSIST09/82/graph/k_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_k':
        with open('./data/ASSIST09/82/graph/e_from_k.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
def normalize_sym(adj):
    """用于对称归一化邻接矩阵"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_row(mx):
    """用于行归一化稀疏矩阵，它接收一个Scipy稀疏矩阵sparse_mx作为输入，并将其转换为PyTorch稀疏张量表示。"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """用于将Scipy稀疏矩阵转换为PyTorch稀疏张量"""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def main(dataset):
    # 从”edges.pkl"文件中加载边信息
    with open('data/ASSIST09/82/edges.pkl', "rb") as f:
        edges = pkl.load(f)
        f.close()

    # 初始化一个数组来储存节点类型
    node_types = np.zeros((edges[0].shape[0],), dtype=np.int32)

    # 从edges[0] 和 edges[2]的行索引中提取唯一值,三类节点
    a = np.arange(60)                                                         #概念节点832 [0：831]
    b = np.arange(60,1747)                                                    #练习节点832 [832:1663]
    c = np.arange(1747,3714)                                                   #学术节点200 [1664,1684]
    print(node_types.shape[0])
    #c = np.arange(1664,1684)                                                   #学生节点20  [1664:1683]
    # 检查节点类型的一致性
    print(a.shape[0], b.shape[0], c.shape[0])
    assert(a.shape[0] + b.shape[0] + c.shape[0] == node_types.shape[0])
    assert(np.unique(np.concatenate((a, b, c))).shape[0] == node_types.shape[0])

    #将节点类型分配给 node_types 数组
    node_types[a.shape[0]:a.shape[0] + b.shape[0]] = 1                          #[832:1663]索引的节点定义为概念节点：表示为1
    node_types[a.shape[0] + b.shape[0]:] = 2                                    #[1664:1684]定义为学生节点:表示为2
    # 检查节点类型计数的总和
    assert(node_types.sum() == b.shape[0] + 2 * c.shape[0])
    # 将 node_types 数组保存到名为 "node_types.npy" 的文件中
    np.save("./data/ASSIST09/55/node_types", node_types)
    
if __name__ == "__main__":
    main("junyi")

