from transformers import PreTrainedModel
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from collections import defaultdict
from torch_scatter import scatter


class CLMessagePassing(MessagePassing):
    def __init__(self, args, channels, dropout):
        super(CLMessagePassing, self).__init__(aggr='add') 
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.dropout = dropout
        self.encode = args.encoder
        self.encoder_layers = args.encoder_layers

        # self.encoder = GCN(in_channels=channels,
        #                      hidden_channels=channels,
        #                      out_channels=channels,
        #                      num_layers=self.encoder_layers,
        #                      dropout=self.dropout,
        #                     )
            
        self.w1 = nn.Linear(channels, channels)
        self.w2 = nn.Linear(channels, channels)
        self.gate = nn.Linear(2 * channels, 1)

        # self.w3 = nn.Linear(channels, channels)
        # self.w4 = nn.Linear(channels, channels)
        # self.gate2 = nn.Linear(2 * channels, 1)

        nn.init.xavier_normal_(self.w1.weight, gain=1.414)
        nn.init.xavier_normal_(self.w2.weight, gain=1.414)
        # nn.init.xavier_normal_(self.w3.weight, gain=1.414)
        # nn.init.xavier_normal_(self.w4.weight, gain=1.414)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        # nn.init.xavier_normal_(self.gate2.weight, gain=1.414)



    # def forward(self, x, node_features, class_features, edge_index, edge_index_full):
    def forward(self, center_features, neighbor_features, edge_index, function):
        if function=='agg_nodes':
            # edge_index is the sub_graph of each nodes
            row, col = edge_index
            num_rows = neighbor_features.shape[0]
            center_features = center_features.clone().unsqueeze(0).expand(num_rows, -1)

            # get weight for each neighbors
            weights_neighbor = self.weight_neighbor(center_features, neighbor_features, row, col)
            out = self.propagate(edge_index, x=neighbor_features, norm=weights_neighbor)
            # print(out.shape)
            agg_embedding = self.alpha * center_features + (1 - self.alpha) * out
            
        if function=="agg_class":
            # edge_index is the structures of the original graph
            # center_neibors = self.encoder(center_features, edge_index)
            norms = torch.ones(edge_index[0].shape[0]).unsqueeze(1).cuda()
            h = self.propagate(edge_index, x=center_features, norm=norms)
            h = self.beta * center_features + (1 - self.beta) * h

            row = torch.tensor([i for i in range(center_features.shape[0])]).cuda()
            row = row.repeat(1, neighbor_features.shape[0]).sort()[0].squeeze(0)
            
            col = torch.tensor([i for i in range(neighbor_features.shape[0])]).cuda()
            col = col.repeat(1,center_features.shape[0]).squeeze(0)
            
            edge_index_new = torch.cat((row.view(1, -1), col.view(1, -1)), dim=0).cuda()

            weights_class = self.weight_neighbor(h, neighbor_features, row, col)
            out = self.propagate(edge_index_new, x=neighbor_features, norm=weights_class)
            agg_embedding = center_features + self.gamma*out
            # print(agg_embedding.shape)
        return agg_embedding
    
    def propagate(self, edge_index, x, norm):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
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

        # 将 defaultdict 转换为字典
        index_groups_dict = dict(index_groups)
        for i in range(len(index_groups_dict)):
            index= torch.tensor(index_groups_dict[i]).cuda()
            w[index] = F.softmax(w[index], dim=0)
        
        return w


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)
    


# class MessagePassing(MessagePassing):
#     def __init__(self, args, channels, dropout):
#         super(MessagePassing, self).__init__(aggr='add') 
#         self.alpha = args.alpha
#         self.dropout = dropout
#         self.encode = args.encoder
#         self.encoder_layers = args.encoder_layers
#         if self.encode:
#             self.encoder = GCN(in_channels=channels,
#                              hidden_channels=channels,
#                              out_channels=channels,
#                              num_layers=self.encoder_layers,
#                              dropout=self.dropout,
#                             )
#         self.w1 = nn.Linear(channels, channels)
#         self.w2 = nn.Linear(channels, channels)
#         self.gate = nn.Linear(2 * channels, 1)
#         nn.init.xavier_normal_(self.w1.weight, gain=1.414)
#         nn.init.xavier_normal_(self.w2.weight, gain=1.414)
#         nn.init.xavier_normal_(self.gate.weight, gain=1.414)


#     def forward(self, prompt_features, node_features, edge_index, edge_index_full):
#         # class_center = x
#         # center_neibors = self.encoder()
#         # if self.encode:
#         #     x = self.encoder(x, edge_index_full)
#         num_rows = node_features.shape[0]
#         classes = prompt_features.clone().unsqueeze(0).expand(num_rows, -1)
#         row, col = edge_index
#         # deg = degree(col, x.size(0), dtype=x.dtype)
#         # deg_inv_sqrt = deg.pow(-0.5)
#         # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#         # print(norm.view(-1, 1).shape)
#         w = self.weight(prompt_features, node_features, row, col)
#         out = self.propagate(edge_index, x=node_features, norm=w)
#         # neighbor = self.message(x[col], norm=a)
#         return self.alpha * classes + (1 - self.alpha) * out

#     def message(self, x_j, norm):
#         # h = torch.cat([x_i, x_j], dim=1)
#         # g = torch.relu(torch.tanh(self.gate(h)))
#         # a = torch.mul(norm.view(-1, 1), g)
#         return norm * x_j  
    
#     def weight(self, prompt_features, node_features, row, col):
#         x_j = self.w2(node_features[col])
#         x_j = F.dropout(x_j, self.dropout, training=self.training)
        
#         num_rows = x_j.shape[0]
#         classes = prompt_features.clone().unsqueeze(0).expand(num_rows, -1)
#         x_i = self.w1(classes)
#         x_i = F.dropout(x_i, self.dropout, training=self.training)
#         h = torch.cat([x_i, x_j], dim=1)

#         # w = torch.relu(torch.tanh(self.gate(h)))
#         w = self.gate(h)
#         w = self.att(w, row)
#         # w = F.dropout(w, self.dropout, training=self.training)
#         # a = torch.mul(norm.view(-1, 1), w)
#         return w
    
#     def att(self, w, row):   
#         index_groups = defaultdict(list)

#         for idx, value in enumerate(row):
#             index_groups[value.item()].append(idx)

#         # 将 defaultdict 转换为字典
#         index_groups_dict = dict(index_groups)
#         for i in range(len(index_groups_dict)):
#             index= torch.tensor(index_groups_dict[i]).cuda()
#             w[index] = F.softmax(w[index], dim=0)
        
#         return w
    
    

class BertGNN(PreTrainedModel):
    def __init__(self, args, data, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.data = data
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        # self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        # print(hidden_dim)
        self.classifier = GCN(in_channels=hidden_dim,
                             hidden_channels=args.hidden_dim,
                             out_channels=n_labels,
                             num_layers=args.num_layers,
                             dropout=args.dropout,
                            )
        # init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):

        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = self.dropout(outputs['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        # print(cls_token_emb.shape)
        logits = self.classifier(cls_token_emb, self.data.edge_index)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
    


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # return x.log_softmax(dim=-1)
        return x



class BertClaInfGNN(PreTrainedModel):
    def __init__(self, data, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.data = data
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')
        
    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):

        # Extract outputs from the model
        bert_outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict,
                                                         output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = bert_outputs['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb, self.data.edge_index)

        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
    

    