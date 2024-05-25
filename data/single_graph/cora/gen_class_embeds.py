from sentence_transformers import SentenceTransformer
import torch
import os
import pandas as pd
from torch_geometric.data import Data
import torch_geometric as pyg


def sbert(device):
    model = SentenceTransformer('LGraphPrompt3/llm_all-MiniLM-L6-v2').to(device)
    return model 

sbert_model = sbert('cuda')
path = 'LGraphPrompt3/data/single_graph/cora/cora.pt'
data = torch.load(path)
# Data(raw_text=[2708], y=[2708], label_names=[7], edge_index=[2, 10858], train_masks=[10], val_masks=[10], test_masks=[10], 
# x=[2708, 384], raw_texts=[2708], category_names=[2708]).

# nx_g = pyg.utils.to_networkx(data, to_undirected=True)
# edge_index = torch.tensor(list(nx_g.edges())).T
# print(edge_index.shape)
# print(data.x)
# print(data.edge_index)
# print(ok)

category_desc = pd.read_csv(
    os.path.join("LGraphPrompt2/data/single_graph/cora/categories.csv"), sep=","
    ).values
label_names = data.label_names

label_desc = []
for i, label in enumerate(label_names):
    # true_ind = label == category_desc[:, 0]
    des = category_desc[i, 1]
    # print(des)
    label_desc.append(des)

# print(label_desc)
sbert_embeds = sbert_model.encode(label_desc, batch_size=8, show_progress_bar=True)
categery_des = torch.tensor(sbert_embeds)

data = Data()
data.class_embeddings = categery_des
data.class_names = label_names
# torch.save(data, 'LGraphPrompt2/data/single_graph/cora/cora_class.pt')

# classes = torch.load('LGraphPrompt2/data/single_graph/cora/cora_class.pt')
# print(classes.class_embeddings)
# print(classes.class_names)