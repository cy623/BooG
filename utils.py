import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_data(dataset_names):
    names = dataset_names
    if names=='cora':
        data = torch.load('TargetGraph/data/single_graph/cora/cora.pt')

    if names=='pubmed':
        data = torch.load('TargetGraph/data/single_graph/pubmed/pubmed.pt')

    if names=='arxiv':
        data = torch.load('TargetGraph/data/single_graph/ogbn-arxiv/arxiv_sbert.pt')

    return data

def load_data_class(dataset_names):
    names = dataset_names
    if names=='cora':
        data = torch.load('TargetGraph/data/single_graph/cora/cora_class.pt')

    if names=='pubmed':
        data = torch.load('TargetGraph/data/single_graph/pubmed/pubmed_class.pt')

    if names=='arxiv':
        data = torch.load('TargetGraph/data/single_graph/ogbn-arxiv/arxiv_class.pt')

    return data


def accuracy(logits, labels):
    # _, indices = torch.max(logits, dim=1)
    correct = torch.sum(logits == labels)
    return correct.item() * 1.0 / len(labels)

def accuracy2(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

# def accuracy(output, labels):
#     preds = output.max(1).type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)


def parse_cora():
    path = 'LGraphPrompt/text_data/Cora/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, np.unique(data_edges, axis=0).transpose()



def parse_pubmed():
    path = 'LGraphPrompt/text_data/PubMed/data/'

    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(path + 'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - \
                1  # subtract 1 to zero-count
            data_Y[i] = label

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_X, data_Y, np.unique(data_edges, axis=0).transpose()


def split_id(split, dataset_str):
    splits_file_path = 'TargetGraph/splits/' + dataset_str + \
                '_split_0.6_0.2_' + str(split) + '.npz'

    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    idx_train = list(np.where(train_mask == 1)[0])
    idx_val = list(np.where(val_mask == 1)[0])
    idx_test = list(np.where(test_mask == 1)[0])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_val, idx_test


def get_idx_split(num_nodes):
    node_id = np.arange(num_nodes)
    np.random.shuffle(node_id)

    train_id = np.sort(node_id[:int(num_nodes * 0.6)])
    val_id = np.sort(
        node_id[int(num_nodes * 0.6):int(num_nodes * 0.8)])
    test_id = np.sort(node_id[int(num_nodes * 0.8):])

    train_mask = torch.tensor(
        [x in train_id for x in range(num_nodes)])
    val_mask = torch.tensor(
        [x in val_id for x in range(num_nodes)])
    test_mask = torch.tensor(
        [x in test_id for x in range(num_nodes)])
    
    idx_train = train_mask.nonzero().squeeze().tolist()
    idx_val = val_mask.nonzero().squeeze().tolist()
    idx_test = test_mask.nonzero().squeeze().tolist()
    return idx_train, idx_val, idx_test

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}

def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file

def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise




class Evaluator:
    def __init__(self, name):
        self.name = name

    def eval(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}
    
def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

