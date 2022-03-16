import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(Embedding, self).__init__()
        self.device = parameter['device']
        self.ent2id = dataset['ent2id']
        self.rel2id = dataset['rel2id']
        self.es = parameter['embed_dim']

        num_ent = len(self.ent2id)
        self.embedding = nn.Embedding(num_ent, self.es)

        num_rel = len(self.rel2id)
        self.embeddingR = nn.Embedding(num_rel, self.es)

        if parameter['data_form'] == 'Pre-Train':
            self.ent2emb = dataset['ent2emb']
            self.embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))
            # self.rel2emb = dataset['rel2emb']
            # self.embeddingR.weight.data.copy_(torch.from_numpy(self.rel2emb))
            nn.init.xavier_uniform_(self.embeddingR.weight)
        elif parameter['data_form'] in ['In-Train', 'Discard']:
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.xavier_uniform_(self.embeddingR.weight)

    def forward(self, triples):
        idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        idxR = [[[self.rel2id[batch[0][1]]]] for batch in triples]
        idxR = torch.LongTensor(idxR).to(self.device)
        return [self.embedding(idx), self.embeddingR(idxR)]



