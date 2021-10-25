import os
import pickle
import networkx as nx
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
from scipy import stats
from torch_geometric.data import Data
import torch.nn.functional as F


GRAPH_EXTENSIONS = [
    '.gpickle',
]

PROCESSED_EXTENSIONS = [
    '.pt',
]

### global defintion
quantile_level = 10
node_attr_onehot_classnum = [2, 6, 10, 10, 11] # node_type, bldg_shape, posx, posy, bldg_area
edge_attr_onehot_classnum = [10, 4] #edge_dist, edge_type




def is_graph_file(filename): 
    return any(filename.endswith(extension) for extension in GRAPH_EXTENSIONS)

def is_processed_file(filename):
    return any(filename.endswith(extension) for extension in PROCESSED_EXTENSIONS)


def get_node_attribute(g, keys, dtype, default = None):
    attri = list(nx.get_node_attributes(g, keys).items())
    attri = np.array(attri)
    attri = attri[:,1]
    if attri[0] == None: 
        attri[:4] = default
    attri = np.array(attri, dtype = dtype)
    return attri

def get_edge_attribute(g, keys, dtype, default = None):
    attri = list(nx.get_edge_attributes(g, keys).items())
    attri = np.array(attri)
    attri = attri[:,1]
    attri = np.array(attri, dtype = dtype)
    return attri


def graph2vector(g):
    # getting edge list, getting node attributes, getting edge attributes
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()
    
    bldg_area = get_node_attribute(g, 'bldg_area', np.int_,  quantile_level + 1)
    bldg_shape = get_node_attribute(g, 'bldg_shape', np.int_, 5) # road is 5
    posx = get_node_attribute(g, 'posx', np.double)
    posy = get_node_attribute(g, 'posy', np.double)

    node_type = np.ones(num_nodes, dtype=np.int_)
    node_type[:4] = 0

    edge_list = np.array(list(g.edges()), dtype=np.int_)
    edge_list = np.transpose(edge_list)

    edge_dist = get_edge_attribute(g, 'edge_dist', np.double)
    edge_type = get_edge_attribute(g, 'edge_type', np.int_)

    node_attr = np.stack((node_type, bldg_shape, posx, posy, bldg_area), 1)
    edge_attr = np.stack((edge_dist, edge_type), 1)
    
    return node_attr, edge_list, edge_attr
 


def transform_to_quantile(arr, level, except_idx = None):
    # res = np.vectorize(lambda x: stats.percentileofscore(arr, x))(arr)
    rang = 1.0 / np.float32(level)
    if except_idx != None:
        bins = np.nanquantile(arr[except_idx:], np.arange(rang,1+rang,rang))
        arr[except_idx:] = np.digitize(arr[except_idx:], bins)
        res = arr
    else:
        bins = np.nanquantile(arr, np.arange(rang,1+rang,rang))    
        res = np.digitize(arr, bins)
    return res



def graph_transform(data):
    node_attr = data.node_attr
    edge_attr = data.edge_attr
    edge_idx = data.edge_index

    select_y = np.int_(np.random.choice(np.arange(4, node_attr.shape[0] - 0.9, 1), 1)[0])
    y = node_attr[select_y, :]
    node_attr = np.delete(node_attr, select_y, 0)

    for i in range(edge_idx.shape[1]):
        if (data.edge_index[0, i] == select_y) or (data.edge_index[1, i] == select_y):
            print(i)
            edge_idx = np.delete(edge_idx, i, 1)
            edge_attr = np.delete(edge_attr, i, 0)
            ##### can also keep edges connected to y
    
    node_attr = torch.tensor(node_attr, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    edge_idx = torch.tensor(edge_idx, dtype=torch.long)

    for i in range(node_attr.shape[1]):
        if i == 0:
            node_attr_onehot = F.one_hot(node_attr[:, i], num_classes=node_attr_onehot_classnum[i])
        else:
            node_attr_onehot = torch.cat( (node_attr_onehot, F.one_hot(node_attr[:, 1], num_classes=node_attr_onehot_classnum[i])), 1 )

    for i in range(edge_attr.shape[1]):
        if i == 0:
            edge_attr_onehot = F.one_hot(edge_attr[:, i], num_classes=edge_attr_onehot_classnum[i])
        else:
            edge_attr_onehot = torch.cat( (edge_attr_onehot, F.one_hot(edge_attr[:, 1], num_classes=edge_attr_onehot_classnum[i])), 1 )

    trans_data = Data(node_attr=node_attr_onehot, edge_attr=edge_attr_onehot,edge_idx = edge_idx, y=y)
    return trans_data

    # get random node, remove it from graph, make it as ground truth
    # probably also save edge info for next step
    # self.transform() to do augmentation




class UrbanGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        raw_graph_dir = []
        for root, _, fnames in sorted(os.walk(self.raw_dir)):
            for fname in fnames:
                if is_graph_file(fname):
                    path = os.path.join(root, fname)
                    raw_graph_dir.append(path)          
        return raw_graph_dir


    @property
    def processed_file_names(self):
        processed_graph_dir = []
        for root, _, fnames in sorted(os.walk(self.processed_dir)):
            for fname in fnames:
                if is_processed_file(fname):
                    path = os.path.join(root, fname)
                    processed_graph_dir.append(path)          
        return processed_graph_dir


    # def download(self):
    #     # Download to `self.raw_dir`.
    #     path = download_url(url, self.raw_dir)
    #     ...


    def process(self):
        self.dataset_attribute_discretize() # self.root inherited from BaseDataset


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # called by get_item() in BaseDatset, after get(), base dataset will implement transform()
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


    def dataset_attribute_discretize(self):
        edge_attr_all = None
        node_attr_all = None
        node_num_list = []
        edge_num_list = []
        list_of_edge_list = []

        i = 0
        # get all values from the entire dataset, store in arrays
        for raw_path in self.raw_paths:
            tmp_graph = nx.read_gpickle(raw_path)
            node_attr, edge_list , edge_attr = graph2vector(tmp_graph)
            node_num_list.append(tmp_graph.number_of_nodes())
            edge_num_list.append(tmp_graph.number_of_edges())
            list_of_edge_list.append(edge_list)

            if i == 0:
                edge_attr_all = edge_attr
                node_attr_all = node_attr
            else:
                edge_attr_all = np.concatenate((edge_attr_all, edge_attr), axis = 0)
                node_attr_all = np.concatenate((node_attr_all, node_attr), axis = 0)
            i += 1


        # quantile contiguous edge_dist, bldg_area, x and y coordinates, to discretization 
        edge_attr_all[:,0] = transform_to_quantile(edge_attr_all[:,0], quantile_level)

        node_attr_all[:,2] = transform_to_quantile(node_attr_all[:,2], quantile_level)
        node_attr_all[:,3] = transform_to_quantile(node_attr_all[:,3], quantile_level)
        ###### raod nodes don't have bldg area, so except the first 4 items
        node_attr_all[:,4] = transform_to_quantile(node_attr_all[:,4], quantile_level, 4) 


        # store quantilized array into .pt files for later using
        curr_node_idx = 0
        curr_edge_idx = 0
        for j in range(len(self.raw_paths)):
            node_vol = node_num_list[j]
            edge_vol = edge_num_list[j]

            curr_node_attr = node_attr_all[curr_node_idx : curr_node_idx+node_vol, :]
            curr_edge_attr = edge_attr_all[curr_edge_idx : curr_edge_idx+edge_vol, :]
            curr_node_idx += node_vol
            curr_edge_idx += edge_vol

            edge_index = list_of_edge_list[j]

            data = Data(node_attr = curr_node_attr, edge_attr = curr_edge_attr, edge_index=edge_index)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(j)))

            #numpy.quantile
            #then input into processed_dir, index from 0 to n-1.





# root = os.getcwd()
# a = UrbanGraphDataset(os.path.join(root,'dataset'))
# a.process()
# a.get(0)
# print(a.raw_file_names)