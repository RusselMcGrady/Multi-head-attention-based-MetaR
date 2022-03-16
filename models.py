from __future__ import print_function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from embedding import *
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from protonet import ProtoNet

class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size

        self.lstm1 = nn.LSTM(input_size=2 * embed_size, hidden_size=2 * out_size, num_layers=1, batch_first=True,
                             bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=2 * embed_size, hidden_size=2 * out_size, num_layers=1, batch_first=True,
                             bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=4 * embed_size, hidden_size=2 * out_size, num_layers=1, batch_first=True,
                             bidirectional=False)
        self.lstm4 = nn.LSTM(input_size=4 * embed_size, hidden_size=1 * out_size, num_layers=1, batch_first=True,
                             bidirectional=False)

        self.label_attn1 = multihead_attention(2 * embed_size, num_heads=5, dropout_rate=0.5)  ## wiki=100 nell=200
        self.label_attn2 = multihead_attention(2 * embed_size, num_heads=5, dropout_rate=0.5)  ## wiki=100 nell=200
        self.droplstm = nn.Dropout(0.5)
        # self.gpu = torch.cuda.is_available()
        # if self.gpu:
        #     self.lstm1 = self.lstm1.cuda()
        #     self.lstm2 = self.lstm2.cuda()
        #     self.lstm3 = self.lstm3.cuda()
        #     self.lstm4 = self.lstm4.cuda()
        #     self.label_attn1 = self.label_attn1.cuda()
        #     self.label_attn2 = self.label_attn2.cuda()

    def forward(self, inputs, label_embs):
        if not hasattr(self, '_flattened'):
            self.lstm1.flatten_parameters()
            self.lstm2.flatten_parameters()
            self.lstm3.flatten_parameters()
            self.lstm4.flatten_parameters()
            setattr(self, '_flattened', True)

        size_s = inputs.shape
        size_q = label_embs.shape
        x = inputs.contiguous().view(size_s[0], size_s[1], -1)
        y = label_embs.contiguous().view(size_q[0], size_q[1], -1)
        hidden, hidden_y = None, None

        x, _ = self.lstm1(x)
        y, _ = self.lstm2(y)
        x = self.droplstm(x)
        y = self.droplstm(y)

        label_attention_output1 = self.label_attn1(x, y, y)
        x = torch.cat([x, label_attention_output1], -1)

        x, _ = self.lstm3(x)

        label_attention_output2 = self.label_attn2(x, y, y)
        x = torch.cat([x, label_attention_output2], -1)

        x, _ = self.lstm4(x)

        x = torch.mean(x, 1)

        return x.view(size_s[0], 1, 1, self.out_size)

class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r[0] - t, 2, -1).squeeze(2) - 0.01 * torch.norm(r[0] - r[1], 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.embedding = Embedding(dataset, parameter)

        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=50, num_hidden1=250,
                                                        num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)

        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.rel_q_sharingR = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t)[0] for t in task]
        supportR, _, _, _ = [self.embedding(t)[1] for t in task]

        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        rel = self.relation_learner(support, support_negative)

        rel.retain_grad()
        supportR.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few + num_sn, -1, -1)
        supportR_s = supportR.expand(-1, few+num_sn, -1, -1)

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
            supportR_q = self.rel_q_sharingR[curr_rel]
        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, [rel_s, supportR_s], few)

                y = torch.Tensor([1]).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                rel_q = rel - self.beta * grad_meta
                supportR_q = supportR - self.beta * grad_meta
            else:
                rel_q = rel
                supportR_q = supportR

            self.rel_q_sharing[curr_rel] = rel_q
            self.rel_q_sharingR[curr_rel] = supportR_q

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)
        queryR_q = supportR_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, [rel_q, queryR_q], num_q)

        return p_score, n_score

class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0, gpu=True, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        #if self.gpu:
            #self.Q_proj = self.Q_proj.cuda()
            #self.K_proj = self.K_proj.cuda()
            #self.V_proj = self.V_proj.cuda()


        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values,last_layer = False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        if last_layer == True:
            return outputs
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries

        return outputs
