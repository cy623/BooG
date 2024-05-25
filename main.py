import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from utils import *
# from model import *
import warnings
import torch.optim as optim
from info_nce import InfoNCE
import json
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import remove_self_loops
from collections import defaultdict
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from sklearn.decomposition import PCA


warnings.filterwarnings('ignore')


class CLMessagePassing(MessagePassing):
    def __init__(self, args, channels, dropout):
        super(CLMessagePassing, self).__init__(aggr='add') 
        self.beta = args.beta
        self.dropout = dropout
        self.w1 = nn.Linear(channels, channels)
        self.w2 = nn.Linear(channels, channels)
        self.gate = nn.Linear(2 * channels, 1)

        self.w3 = nn.Linear(channels, channels)

        nn.init.xavier_normal_(self.w1.weight, gain=1.414)
        nn.init.xavier_normal_(self.w2.weight, gain=1.414)
        nn.init.xavier_normal_(self.w3.weight, gain=1.414)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)


    def forward(self, center_features, sub_edges_index, node_features):
        row, col = sub_edges_index
        weights_neighbor = self.weight_neighbor(center_features, node_features, row, col)
        out = self.propagate(sub_edges_index, x=node_features, norm=weights_neighbor)
        agg_embedding = self.beta * center_features + (1 - self.beta) * out
        agg_embedding = self.w3(agg_embedding)

        return agg_embedding
    
    def propagate(self, edge_index, x, norm):
        row, col = edge_index
        x_j = x[col]
        out = self.message(x_j, norm)
        out = self.aggregate(out, row)
        out = self.update(out)
        
        return out

    def message(self, x_j, norm):
        return norm * x_j 

    def aggregate(self, x, index):   
        aggr_out = scatter(x, index, dim=0, reduce='sum')
        return aggr_out 
    
    def update(self, aggr_out):
        return aggr_out
    
    def weight_neighbor(self, centers, neighbors, row, col):
        x_i = self.w1(centers[row])
        x_j = self.w2(neighbors[col])

        h = torch.cat([x_i, x_j], dim=1)
        w = self.gate(h)
        # get attentions for each sub_graph
        w = self.att(w, row)

        w = F.dropout(w, self.dropout, training=self.training)

        return w 

    def att(self, w, row):   
        # the same row index is a group, get att by softmax
        index_groups = defaultdict(list)

        for idx, value in enumerate(row):
            index_groups[value.item()].append(idx)

        index_groups_dict = dict(index_groups)
        for i in range(len(index_groups_dict)):
            index= torch.tensor(index_groups_dict[i]).cuda()
            w[index] = F.softmax(w[index], dim=0)
        
        return w



class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_class):
        super(Classifier, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.dropout(out, self.dropout, training=self.training)
        return F.log_softmax(out, dim=1)
    

# def tsne(X, Y, label_names, path):
#     X= X.cpu().numpy()
#     Y= Y.cpu().numpy()

#     tsne = TSNE(n_components=2, random_state=0)
#     X_tsne = tsne.fit_transform(X)
#     plt.figure(figsize=(8, 6))
#     for i in range(len(label_names)):
#         plt.scatter(X_tsne[Y == i, 0], X_tsne[Y == i, 1], label=label_names[i], marker='o')

#     plt.legend().set_visible(False)
#     plt.savefig(path, bbox_inches='tight')
#     plt.show()


def get_sub_edges_index(neighbor_nodes):
    sub_edges_index = torch.empty(2,0).cuda()
    for i in range(len(neighbor_nodes)):
        l = neighbor_nodes[i].shape[0]
        value = i
        source_nodes = torch.full((l,), value).cuda()
        edge_index = torch.stack([source_nodes, neighbor_nodes[i]], dim=0)
        sub_edges_index = torch.cat((sub_edges_index, edge_index), dim=1)
        
    sub_edges_index = torch.tensor(sub_edges_index, dtype=torch.int)
    return sub_edges_index


def sim(matrix1, matrix2):
    matrix_1 = torch.linalg.norm(matrix1, axis=1)
    matrix_2 = torch.linalg.norm(matrix2, axis=1)
    dot_product = torch.matmul(matrix1, matrix2.T)
    cosine_similarity = dot_product / torch.outer(matrix_1, matrix_2)

    return cosine_similarity


def accuracy(logits, labels):
    # _, indices = torch.max(logits, dim=1)
    correct = torch.sum(logits == labels)
    return correct.item() * 1.0 / len(labels)


def classifier_accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def k_shot(data, k, idx_train):
    y = data.y
    num_classes = data.num_classes  
    selected_indices = []  
    
    for class_label in range(num_classes):
        class_indices = torch.where(y[idx_train] == class_label)[0]  
        selected_indices.extend(class_indices[:k].cpu().numpy())  
    selected_indices = torch.tensor(selected_indices)
    return selected_indices


def pretrain(args, data, class_data, idx_test, state='pretrain'):
    num_nodes = data.num_nodes 
    edge_index = data.edge_index
    num_class = data.num_classes
    channels = data.x.shape[1]

    # all_sub_graph = []
    neighbor_nodes = []
    for i in range(num_nodes):
        sub_graph = k_hop_subgraph(i, edge_index=edge_index, num_hops=args.hops)
        # all_sub_graph.append(sub_graph)
        neighbor_nodes.append(sub_graph[0])
    sub_edges_index = get_sub_edges_index(neighbor_nodes)
    sub_edges_index = remove_self_loops(sub_edges_index)[0]
    sub_edges_index = sub_edges_index.to(torch.int64)


    node_features = data.x
    class_features = class_data.class_embeddings

    center_features = []
    for _ in range(num_class):
        center_feature = node_features + args.alpha*class_features[_]
        center_features.append(center_feature)

            
    if state == 'pretrain':
        model = CLMessagePassing(args, channels, args.dropout).cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_function = InfoNCE(temperature=args.t, negative_mode='paired')


        for epoch in range(args.pre_epochs):
            model.train()
            optimizer.zero_grad()

            output_set = []
            for i in range(num_class):
                output = model(center_features[i], sub_edges_index, node_features)
                output_set.append(output)
            
            loss = 0
            for j in range(num_class):
                query = output_set[j]
                positive_keys = center_features[j]
                negative_keys = torch.stack([element for idx, element in enumerate(output_set) if idx not in [j]])
                negative_keys = torch.transpose(negative_keys, 0,1)
                loss_infonce = loss_function(query, positive_keys, negative_keys)
                loss += loss_infonce
            loss = (1/num_class)*loss

            loss.backward()
            optimizer.step()

            print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss.item()),
                )
        
        # para = model.state_dict()
        # torch.save(para, 'TargetGraph/para/'+args.dataset+'.pth')
    if state=='eval':
        model = CLMessagePassing(args, channels, args.dropout).cuda()
        model.load_state_dict(torch.load('TargetGraph/para/cora.pth'))

    model.eval()

    sim = []
    emb_set = []
    for i in range(num_class):
        output = model(center_features[i], sub_edges_index, node_features)
        emb_set.append(output.detach())
        cosine_sim = F.cosine_similarity(class_features[i], output).unsqueeze(1)
        sim.append(cosine_sim)
    sim = torch.cat(sim, dim=1)
    pre =  torch.argmax(sim, dim=1)
    pretrain_acc = accuracy(pre[idx_test], data.y[idx_test])
    print('pretrain_acc', pretrain_acc)

    node_emb = []
    for i in range(num_nodes):
        ind = pre[i]
        embs = emb_set[ind][i].unsqueeze(0)
        node_emb.append(embs)
    node_emb = torch.cat(node_emb, dim=0)
    return node_emb, pretrain_acc
    # return emb_set, pretrain_acc


def prompt_tune(args, data, class_data, node_emb, idx_train, idx_val, idx_test):
    prompt_node = class_data.class_embeddings
    pretrain_features = node_emb # torch.cat(node_emb,dim=1)
    # pretrain_features = torch.cat(node_emb,dim=1)
    num_class = data.num_classes
    num_nodes = data.num_nodes 
    channels = data.x.shape[1] # *num_class
    # channels = data.x.shape[1]*num_class
    labels  = data.y


    classifier = Classifier(channels, args.hidden, args.dropout, num_class).cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr_prompt, weight_decay=args.weight_decay_prompt)

    
    best_val_loss = float('inf')
    acc = float('-inf')
    for epoch in range(args.epochs):
        classifier.train()
        optimizer.zero_grad()
 
        output = classifier(pretrain_features)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = classifier_accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        classifier.eval()

           
        output = classifier(pretrain_features)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = classifier_accuracy(output[idx_val], labels[idx_val])

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = classifier_accuracy(output[idx_test], labels[idx_test])

        print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train),
                'loss_test: {:.4f}'.format(loss_test.item()),
                'acc_test: {:.4f}'.format(acc_test),
            #      'time: {:.4f}s'.format(time.time() - t)
        )
          
        if loss_val <= best_val_loss: 
            best_val_loss = loss_val
            if acc_test > acc:
                acc = acc_test
                test_loss = loss_test

    print("GNN Test set results:",
        "loss= {:.4f}".format(test_loss.item()),
        "accuracy= {:.4f}".format(acc))      

    return acc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--type', type=str, default='super', help='training types')
    parser.add_argument('--k', type=int, default=5, help='k shot')
    parser.add_argument('--dataset', type=str, default='cora', help='Name of dataset')
    parser.add_argument('--hidden', type=int, default=128, help='')
    parser.add_argument('--pre_epochs', type=int, default=20, help='Maximal number of epochs.')
    parser.add_argument('--epochs', type=int, default=500, help='Maximal number of epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate of lm.')
    parser.add_argument('--lr_prompt', type=float, default=0.01, help='Initial learning rate of lm.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--weight_decay_prompt', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

    parser.add_argument('--hops', type=int, default=2, help='Number of gnn layers')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--t', type=float, default=0.1)
    args = parser.parse_args()

    print(torch.cuda.is_available())
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + '0') if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # data
    # Data(raw_text=[2708], y=[2708], label_names=[7], edge_index=[2, 10858], 
    # train_masks=[10], val_masks=[10], test_masks=[10], x=[2708, 384], raw_texts=[2708], category_names=[2708])
    print('Loading data......')
    data = load_data(args.dataset).to(device)
    # print(data)
    data.num_nodes = len(data.y)
    data.num_classes = len(data.label_names)
    print('Dataset:', args.dataset)
    print("Num_nodes:", data.num_nodes)
    print("Num_classes:", data.num_classes)

    # class data
    # Data(class_embeddings=[7, 384], class_names=[7])
    print('Loading prompt data......')
    class_data = load_data_class(args.dataset).to(device)
    # print(class_data)
    
    preacc = []
    for _ in range(10):
        print('Run:',_)
        split = _
        if args.dataset=='cora' or args.dataset=='pubmed':
            idx_train, idx_val, idx_test = split_id(split, args.dataset)
        else:
            idx_train, idx_val, idx_test = get_idx_split(data.num_nodes)
        
        if args.type != 'super':
              idx_train = k_shot(data, args.k, idx_train)

        node_emb, pretrain_acc = pretrain(args, data, class_data, idx_test, state='pretrain')
        acc = prompt_tune(args, data, class_data, node_emb, idx_train, idx_val, idx_test)
        preacc.append(acc)
        print(pretrain_acc)
    
    print('ave_acc: {:.4f}'.format(np.mean(preacc)), '+/- {:.4f}'.format(np.std(preacc)))

    outfile_name = f"{args.dataset}_type_{args.type}_results.txt"
    Hyperparameters = f"lr{args.lr_prompt}_weight_decay{args.weight_decay_prompt}_drop{args.dropout}"
    print(outfile_name)
    results_dict = {}
    results_dict['test_acc_mean'] = float(np.mean(preacc))
    results_dict['test_acc_std'] = float(np.std(preacc))

    with open(os.path.join('TargetGraph/run/llm+pretrain+prompt/', outfile_name), 'a') as outfile:
        outfile.write(Hyperparameters)
        outfile.write('\n')
        outfile.write(json.dumps(results_dict))
        outfile.write('\n')



if __name__ == "__main__":
    main()