from models import *  ##modelsOri
from tensorboardX import SummaryWriter
import os
import sys
import torch
import shutil
import logging
from KGAT import KGATb
import numpy as np
import pandas as pd
import scipy.sparse as sp


class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        self.all_data_loader = data_loaders[3]
        # parameters
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        # epoch
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        # device
        self.device = parameter['device']

        self.metaR = MetaR(dataset, parameter)
        self.metaR.to(self.device)

        self.embedding = Embedding(dataset, parameter)
        self.embedding.to(self.device)

        self.laplacian_type = parameter['laplacian_type']

        # construct model & optimizer
        self.model = KGATb(self.parameter, 34272, 34272, 649, None, None)
        torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        self.cf_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.kg_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # sample one batch from data_loader
        all_task, curr_rel = self.all_data_loader.next_batch()
        _, _, _, _ = [self.embedding(t) for t in all_task]

        self.train_relation_dict = self.embedding.train_relation_dict
        self.create_adjacency_dict()
        self.create_laplacian_dict()

        # optimizer
        # self.optimizer = torch.optim.Adam(self.metaR.parameters(), self.learning_rate)
        # tensorboard log writer
        if parameter['step'] == 'train':
            self.writer = SummaryWriter(os.path.join(parameter['log_dir'], parameter['prefix']))
        # dir
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        logging_dir = os.path.join(self.parameter['log_dir'], self.parameter['prefix'], 'res.log')
        logging.basicConfig(filename=logging_dir, level=logging.INFO, format="%(asctime)s - %(message)s")

        # load state_dict and params
        if parameter['step'] in ['test', 'dev']:
            self.reload()

    def reload(self):
        if self.parameter['eval_ckpt'] is not None:
            state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
        else:
            state_dict_file = os.path.join(self.state_dir, 'state_dict')
        self.state_dict_file = state_dict_file
        logging.info('Reload state_dict from {}'.format(state_dict_file))
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.metaR.load_state_dict(state)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(self.metaR.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict'))

    def write_training_log(self, data, epoch):
        self.writer.add_scalar('Training_Loss', data['Loss'], epoch)

    def write_validating_log(self, data, epoch):
        self.writer.add_scalar('Validating_MRR', data['MRR'], epoch)
        self.writer.add_scalar('Validating_Hits_10', data['Hits@10'], epoch)
        self.writer.add_scalar('Validating_Hits_5', data['Hits@5'], epoch)
        self.writer.add_scalar('Validating_Hits_1', data['Hits@1'], epoch)

    def logging_training_data(self, data, epoch):
        logging.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path, istest=False):
        setname = 'dev set'
        if istest:
            setname = 'test set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank
        if 8 < rank <= 10:   ##NPS detractor
            data['NPS'] += 1

    def do_one_step(self, task, iseval=False, curr_rel=''):
        loss, p_score, n_score = 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score = self.metaR(task, iseval, curr_rel)
            y = torch.Tensor([1]).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
            # loss = loss1 + lossR
            loss.backward()
            self.optimizer.step()
        elif curr_rel != '':
            p_score, n_score = self.metaR(task, iseval, curr_rel)
            y = torch.Tensor([1]).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
            # loss = loss1 + lossR
        return loss, p_score, n_score

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(torch.clamp(torch.Tensor(rowsum), min=1e-12).numpy(), -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo()).to(self.device)

    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(68544, 68544))
            self.adjacency_dict[r] = adj

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0
        cf_total_loss = 0
        kg_total_loss = 0

        # training by epoch
        for e in range(self.epoch):
            self.model.train()

            # sample one batch from data_loader
            train_task, curr_rel = self.train_data_loader.next_batch()
            support, support_negative, query, negative = [self.embedding(t) for t in train_task]

            # self.train_relation_dict = self.embedding.train_relation_dict
            # self.create_adjacency_dict()
            # self.create_laplacian_dict()

            cf_batch_loss = self.model(support[0], support[1], support_negative[1], self.A_in, mode='train_cf')
            cf_batch_loss.backward()
            self.cf_optimizer.step()
            self.cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            kg_batch_loss = self.model(support[0], support[2], support[1], support_negative[1], self.A_in, mode='train_kg')
            kg_batch_loss.backward()
            self.kg_optimizer.step()
            self.kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            # total_batch_loss = cf_batch_loss+kg_batch_loss
            # total_batch_loss.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()

            relations = list(self.laplacian_dict.keys())
            self.model(support[0], support[2], support[1], relations, mode='update_att')

            # loss, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel)
            # print the loss on specific epoch
            if e % self.print_epoch == 0:
                loss_num = cf_batch_loss.item()+kg_batch_loss.item()
                self.write_training_log({'Loss': loss_num}, e)
                print("Epoch: {}\tLoss: {:.4f}".format(e, loss_num))
            # save checkpoint on specific epoch
            if e % self.checkpoint_epoch == 0 and e != 0:
                print('Epoch  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)
            # do evaluation on specific epoch
            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} has finished, validating...'.format(e))
                # model.eval()

                valid_data = self.eval(istest=False, epoch=e)
                self.write_validating_log(valid_data, e)

                metric = self.parameter['metric']
                # early stopping checking
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    bad_counts = 0
                    # save current best
                    self.save_checkpoint(best_epoch)
                else:
                    print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                        metric, best_value, best_epoch, bad_counts))
                    bad_counts += 1

                if bad_counts >= self.early_stopping_patience:
                    print('\tEarly stopping at epoch %d' % e)
                    break

        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        self.save_best_state_dict(best_epoch)
        print('Finish')

    def eval(self, model=None, istest=False, epoch=None):
        # self.metaR.eval()
        # clear sharing rel_q
        # self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0, 'NPS': 0}
        ranks = []

        t = 0
        temp = dict()
        cf_total_loss_eval = 0
        kg_total_loss_eval = 0
        while True:
            # sample all the eval tasks
            # sample one batch from data_loader
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1

            support, support_negative, query, negative = [self.embedding(t, iseval=True) for t in eval_task]

            # if not model:
            # self.model.train()
            # self.train_relation_dict = self.embedding.train_relation_dict
            # self.create_adjacency_dict()
            # self.create_laplacian_dict()
            # construct model & optimizer

            # model.train()

            # cf_optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            # kg_optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            cf_batch_loss = self.model(support[0], support[1], support_negative[1], self.A_in, mode='train_cf')
            # cf_batch_loss.backward()
            # cf_optimizer.step()
            # cf_optimizer.zero_grad()
            cf_total_loss_eval += cf_batch_loss.item()

            kg_batch_loss = self.model(support[0], support[2], support[1], support_negative[1], self.A_in, mode='train_kg')
            # kg_batch_loss.backward()
            # kg_optimizer.step()
            # kg_optimizer.zero_grad()
            kg_total_loss_eval += kg_batch_loss.item()

            relations = list(self.laplacian_dict.keys())
            self.model(support[0], support[2], support[1], relations, mode='update_att')

            self.model.eval()

            # eval_task, curr_rel = data_loader.next_one_on_eval()

            # self.metaR.train()  ## RNN must start train before eval
            # _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel)
            with torch.no_grad():
                n_score, p_score = self.model(query[0], query[2], query[1], negative[1], mode='predict_kg')

            # x = batch_scores
            x = torch.cat([n_score, p_score], 0)

            self.rank_predict(data, x, ranks)

            # print current temp data dynamically
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\tNPS: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1'], temp['Hits@5'] - temp['NPS']))
            sys.stdout.flush()

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter['step'] == 'train':
            self.logging_training_data(data, epoch)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)

        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1'], temp['Hits@5'] - temp['NPS']))

        return data

    def eval_by_relation(self, istest=False, epoch=None):
        self.metaR.eval()
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        all_data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        all_t = 0
        all_ranks = []

        for rel in data_loader.all_rels:
            print("rel: {}, num_cands: {}, num_tasks:{}".format(
                rel, len(data_loader.rel2candidates[rel]), len(data_loader.tasks[rel][self.few:])))
            data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
            temp = dict()
            t = 0
            ranks = []
            while True:
                eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
                if eval_task == 'EOT':
                    break
                t += 1

                _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=rel)
                x = torch.cat([n_score, p_score], 1).squeeze()

                self.rank_predict(data, x, ranks)

                for k in data.keys():
                    temp[k] = data[k] / t
                sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
                sys.stdout.flush()

            print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))

            for k in data.keys():
                all_data[k] += data[k]
            all_t += t
            all_ranks.extend(ranks)

        print('Overall')
        for k in all_data.keys():
            all_data[k] = round(all_data[k] / all_t, 3)
        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            all_t, all_data['MRR'], all_data['Hits@10'], all_data['Hits@5'], all_data['Hits@1']))

        return all_data

