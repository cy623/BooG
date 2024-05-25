import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        positive_score = torch.matmul(anchor, positive.T) / self.temperature
        negative_scores = torch.matmul(anchor, negatives.T) / self.temperature

        loss = -torch.log(torch.exp(positive_score) / torch.sum(torch.exp(negative_scores), dim=1))

        return loss.mean()
    

def cosine_similarity_between_groups(group1, group2):
    group1_normalized = F.normalize(group1, dim=1)
    group2_normalized = F.normalize(group2, dim=1)

    similarity = torch.sum(group1_normalized * group2_normalized, dim=1)

    return similarity