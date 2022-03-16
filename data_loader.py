import random
import numpy as np
import logging


class DataLoader(object):
    def __init__(self, dataset, parameter, step='train'):
        self.curr_rel_idx = 0
        self.tasks = dataset[step+'_tasks']
        self.rel2candidates = dataset['rel2candidates']
        self.e1rel_e2 = dataset['e1rel_e2']
        self.all_rels = sorted(list(self.tasks.keys()))
        self.num_rels = len(self.all_rels)
        self.few = parameter['few']
        self.bs = parameter['batch_size']
        self.nq = parameter['num_query']

        if step != 'train':
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few:])
            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0

    def next_one(self):
        # shift curr_rel_idx to 0 after one circle of all relations
        if self.curr_rel_idx % self.num_rels == 0:
            random.shuffle(self.all_rels)
            self.curr_rel_idx = 0

        # get current relation and current candidates
        curr_rel = self.all_rels[self.curr_rel_idx]
        self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels  # shift current relation idx to next
        curr_cand = self.rel2candidates[curr_rel]
        while len(curr_cand) <= 10 or len(self.tasks[curr_rel]) <= 10:  # ignore the small task sets
            curr_rel = self.all_rels[self.curr_rel_idx]
            self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
            curr_cand = self.rel2candidates[curr_rel]

        # get current tasks by curr_rel from all tasks and shuffle it
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few+self.nq)
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]

        # construct support and query negative triples
        support_negative_triples = []
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
            support_negative_triples.append([e1, rel, negative])

        negative_triples = []
        for triple in query_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
            negative_triples.append([e1, rel, negative])

        return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel

    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.bs)]

        support, support_negative, query, negative, curr_rel = zip(*next_batch_all)
        return [support, support_negative, query, negative], curr_rel

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        # get current triple
        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2:
                negative_triples.append([e1, rel, negative])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel

    def next_one_on_eval_by_relation(self, curr_rel):
        if self.curr_tri_idx == len(self.tasks[curr_rel][self.few:]):
            self.curr_tri_idx = 0
            return "EOT", "EOT"

        # get current triple
        query_triple = self.tasks[curr_rel][self.few:][self.curr_tri_idx]
        self.curr_tri_idx += 1
        # curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2:
                negative_triples.append([e1, rel, negative])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel

def train_generate(data, curr_rel, rel2candidates, batch_size, few, symbol2id, ent2id, e1rel_e2):
        logging.info('LOADING TRAINING DATA')
        # train_tasks = json.load(open(dataset + '/train_tasks.json'))
        logging.info('LOADING CANDIDATES')
        # rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
        rel2candidates = rel2candidates
        task_pool = list(curr_rel)
        num_tasks = len(task_pool)
        rel_idx = 0

        for each_rel in curr_rel:
            if rel_idx % num_tasks == 0:
                random.shuffle(task_pool)
            query = task_pool[rel_idx % num_tasks]
            rel_idx += 1
            candidates = rel2candidates[query]

            if len(candidates) <= 20:
                # print 'not enough candidates'
                continue

            train_and_test = data[query]

            random.shuffle(train_and_test)

            support_triples = train_and_test[:few]

            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

            support_left = [ent2id[triple[0]] for triple in support_triples]
            support_right = [ent2id[triple[2]] for triple in support_triples]

            all_test_triples = train_and_test[few:]

            if len(all_test_triples) == 0:
                continue

            if len(all_test_triples) < batch_size:
                query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
            else:
                query_triples = random.sample(all_test_triples, batch_size)

            query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            query_left = [ent2id[triple[0]] for triple in query_triples]
            query_right = [ent2id[triple[2]] for triple in query_triples]

            false_pairs = []
            false_left = []
            false_right = []
            for triple in query_triples:
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)
                    if (noise not in e1rel_e2[e_h + rel]) and noise != e_t:
                        break
                false_pairs.append([symbol2id[e_h], symbol2id[noise]])
                false_left.append(ent2id[e_h])
                false_right.append(ent2id[noise])

            yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right, each_rel
