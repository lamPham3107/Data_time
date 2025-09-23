import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from .template import MetaTemplate
from .bdc_module import BDC
from sklearn.linear_model import LogisticRegression

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(-cum_dists[:, :, l - 1, 0] / lbda) +
            torch.exp(-cum_dists[:, :, l - 1, 1] / lbda) +
            torch.exp(-cum_dists[:, :, l, 0] / lbda)
        )

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(-cum_dists[:, :, l - 1, m - 1] / lbda) +
                torch.exp(-cum_dists[:, :, l, m - 1] / lbda)
            )

        # last column
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(-cum_dists[:, :, l - 1, -2] / lbda) +
            torch.exp(-cum_dists[:, :, l - 1, -1] / lbda) +
            torch.exp(-cum_dists[:, :, l, -2] / lbda)
        )

    return cum_dists[:, :, -1, -1]

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.class_way = 'add'
        self.method = params.method

    def set_forward(self, x, is_feature=False):
        if self.class_way == 'add':
            z_support, z_query, support_feature, query_feature = self.parse_feature(x, is_feature)
            z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
            z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
            scores1 = self.metric(z_query, z_proto)
            return scores1

        if self.class_way == 'score':
            z_support1, z_query1, z_support2, z_query2 = self.parse_feature(x, is_feature)
            z_proto1 = z_support1.contiguous().view(self.n_way, self.n_support, -1).mean(1)
            z_query1 = z_query1.contiguous().view(self.n_way * self.n_query, -1)
            scores1 = self.metric(z_query1, z_proto1)
            z_proto2 = z_support2.contiguous().view(self.n_way, self.n_support, -1).mean(1)
            z_query2 = z_query2.contiguous().view(self.n_way * self.n_query, -1)
            scores2 = self.metric(z_query2, z_proto2)
            scores = scores1 + scores2
            return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        y_label = np.repeat(range(self.n_way), self.n_query)

        y_query = Variable(y_query.cuda())
        y_support = Variable(y_support.cuda())

        x = Variable(x.cuda())
        scores = self.set_forward(x)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def forward_meta_val_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.val_n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.val_n_way), self.n_query)
        scores = self.forward_meta_val(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def metric(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if self.n_support > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)
        return score

    def euclidean_dist(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score
