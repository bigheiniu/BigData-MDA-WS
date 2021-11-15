import pytorch_lightning as pl
import torch
import torch.nn as nn
import pandas as pd
from Util import  evaluation
import numpy as np
from Dataset import CommenetDataset
from Dataset import AdvTextDataset, EANNTextDataset
from Models.TextTrainer import TextClf
from Models.SimpleTrainer import SimpleTrainer
from Models.DomainClf import DomainClf
from itertools import chain
from Models.Layers import ReverseLayerF
from argparse import ArgumentParser
# from Models.l2w import step_l2w_group_net
import math



class FullWeightModel(nn.Module):
    def __init__(self, hparams, n_groups, hidden_size, tgt_domain,cls_emb=256, gw_hidden_dim=768):
        super(FullWeightModel, self).__init__()
        self.n_groups = n_groups
        self.hparams = hparams
        self.cls_emb = cls_emb
        h_dim = gw_hidden_dim
        # self.domain_emb = nn.Embedding(self.n_groups, self.cls_emb)
        print("n groups {} emb {}".format(self.n_groups, self.cls_emb))
        self.domain_emb = nn.Embedding(self.n_groups, self.cls_emb, padding_idx=tgt_domain)
        hidden_size_input = hidden_size + 2 + self.cls_emb
        if self.hparams.is_omit_domain_weight:
            hidden_size_input -= self.cls_emb
        if self.hparams.is_omit_logits:
            hidden_size_input -= 2

        self.ins_weight = nn.Sequential(
            # nn.Linear(hidden_size + self.cls_emb + 2, h_dim),
            nn.Linear(hidden_size_input, h_dim),
            # nn.Linear(hidden_size + 2, h_dim),
            # nn.Linear(hidden_size  + 2, h_dim),
            nn.ReLU(),  # Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            # # nn.ReLU(),  # Tanh(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, tgt_feature, expert_logits, mask):
        '''
        item_loss = 1 is just the placeholder
        '''
        # detach the feature
        # tgt_feature_here = tgt_feature.detach()
        tgt_feature_here = tgt_feature
        tgt_feature_here = tgt_feature_here.unsqueeze(1)
        tgt_feature_here = tgt_feature_here.repeat(1, self.n_groups, 1)
        batch_size = tgt_feature_here.shape[0]
        domain = torch.tensor([list(range(self.n_groups))] * batch_size)
        domain = domain.to(tgt_feature.device)
        domain_weight = self.domain_emb(domain)
        if self.hparams.is_omit_domain_weight is False:
            tgt_feature_here = torch.cat([tgt_feature_here, domain_weight], dim=-1)
        if self.hparams.is_omit_logits is False:
            tgt_feature_here = torch.cat([tgt_feature_here, expert_logits], dim=-1)
        final_weight = self.ins_weight(tgt_feature_here)
        if self.hparams.is_sigmoid_weight:
            final_weight = torch.sigmoid(final_weight)

        return torch.mean(final_weight * expert_logits, dim=1), final_weight


class SimpleGroupWeight(nn.Module):
    def __init__(self, n_groups):
        super(SimpleGroupWeight, self).__init__()
        self.group_count = n_groups
        self.weight = nn.Parameter(torch.FloatTensor(1, n_groups, 1), requires_grad=True)
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.group_count)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, expert_logits):
        # expert_logits [batch_size, group_count, num_class]
        mix_expert_logits = torch.sum(torch.sigmoid(self.weight) * expert_logits, dim=1)
        return mix_expert_logits



domain_map = {"gossip":0,"politi":1, "health_deterrent":2}

class MDAWS(SimpleTrainer):
    def __init__(self, hparams, textClf_ckpt=None):
        super(MDAWS, self).__init__(hparams)
        self.hparams = hparams
        self.domain_adv = DomainClf(hparams)
        self.pre_train_epochs = hparams.pre_train_epochs
        self.TextClf = TextClf(hparams, model_name='cnn')
        self.hyper_lambda = hparams.hyper_lambda
        self.class_count = self.hparams.class_count
        # Multiple Expert
        self.source_groups_count = len(self.hparams.src_domain.split(",")) + 1
        self.classifiers = nn.Linear(self.hparams.hidden_size, self.class_count * self.source_groups_count)
        self.is_group_weight = self.hparams.is_group_weight

        if self.hparams.is_group_weight:
            self.group_weight = SimpleGroupWeight(n_groups=self.source_groups_count)
        else:
            tgt_domain = domain_map[self.hparams.tgt_domain]
            self.group_weight = FullWeightModel(hparams, n_groups=self.source_groups_count, hidden_size=self.hparams.hidden_size,
                                                cls_emb=self.hparams.cls_emb, gw_hidden_dim=self.hparams.gw_hidden_dim, tgt_domain=tgt_domain)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()


    def labeled_feature_distance(self, src_features, src_rumor_y, tgt_features, tgt_rumor_y):
        # minimize the distance between the same items
        unique_tgt_y = set(torch.unique(tgt_rumor_y).cpu().values.tolist())
        unique_src_y = set(torch.unique(src_rumor_y).cpu().values.tolist())
        intersect_y = unique_src_y.intersection(unique_tgt_y)
        similar_distance = 0
        for y in intersect_y:
            src_features_y = src_features[(src_rumor_y == y).nonzero()]
            src_features_not_y = src_features[(src_rumor_y != y).nonzero()]
            tgt_features_y = tgt_features[(tgt_rumor_y == y).nonzero()]
            tgt_features_not_y = tgt_features[(tgt_rumor_y != y).nonzero()]

            similar_distance += self.distance(src_features_y, tgt_features_y)
            similar_distance -= self.distance(src_features_y, tgt_features_not_y)
            similar_distance -= self.distance(tgt_features_y, src_features_not_y)

        return similar_distance

    def adversarial_loss(self,features, domain_y):
        reverse_features = ReverseLayerF.apply(features, self.hyper_lambda)
        _, domain_loss = self.domain_adv(reverse_features, domain_y)

        return domain_loss

    def first_stage(self, src_x=None, src_domain_y=None,
                               src_rumor_y=None, tgt_x=None, tgt_domain_y=None):
        src_feature = self.TextClf.forward(x=src_x, y=None)[0]
        tgt_feature = self.TextClf.forward(x=tgt_x, y=None)[0]
        src_reverse_features = ReverseLayerF.apply(src_feature, self.hyper_lambda)
        tgt_reverse_features = ReverseLayerF.apply(tgt_feature, self.hyper_lambda)
        # adversarial loss
        # src_domain_y is one-hot vector
        _, src_domain_loss = self.domain_adv(src_reverse_features, src_domain_y)
        _, tgt_domain_loss = self.domain_adv(tgt_reverse_features, tgt_domain_y)
        adv_loss = src_domain_loss + tgt_domain_loss

        # separately train the classifiers
        src_logits = self.classifiers(src_feature)
        # binary classification
        src_logits = src_logits.view(-1, self.source_groups_count, self.class_count)
        # convert the domain_y into one hot
        src_domain_onehot = src_feature.new_zeros(src_feature.shape[0], self.hparams.domain_class)
        src_domain_onehot.zero_()
        if len(src_domain_y.shape) != 2:
            src_domain_y1 = src_domain_y.unsqueeze(1)
        else:
            src_domain_y1 = src_domain_y
        # mask other source classifier
        src_domain_onehot.scatter_(1, src_domain_y1, 1)
        src_logits = torch.sum(src_logits * src_domain_onehot.unsqueeze(-1), dim=1)
        clf_loss = self.cross_entropy_loss(src_logits, src_rumor_y)

        return clf_loss + adv_loss, (clf_loss, src_logits, src_feature), tgt_feature

    def second_stage(self, src_x=None, src_domain_y=None,src_rumor_y=None,
                     tgt_x=None, tgt_domain_y=None, tgt_rumor_y=None,
                     tgt_no_x=None, tgt_no_domain_y=None, tgt_no_rumor_y=None):
        if tgt_no_x is not None:
            # utilize all the target data to train.
            loss_first, _, _ = self.first_stage(src_x, src_domain_y, src_rumor_y, tgt_no_x, tgt_no_domain_y)
            tgt_feature = self.TextClf.forward(x=tgt_x, y=None)[0]
        else:
            loss_first, _, tgt_feature = self.first_stage(src_x, src_domain_y, src_rumor_y, tgt_x, tgt_domain_y)

        expert_logits = self.classifiers(tgt_feature)
        expert_logits = expert_logits.view(-1, self.source_groups_count, self.class_count)
        tgt_domain_onehot = tgt_feature.new_ones(tgt_feature.shape[0], self.hparams.domain_class)
        if len(src_domain_y.shape) != 2:
            tgt_domain_y1 = tgt_domain_y.unsqueeze(1)
        else:
            tgt_domain_y1 = tgt_domain_y

        tgt_domain_onehot.scatter_(1, tgt_domain_y1, 0)
        expert_logits = tgt_domain_onehot.unsqueeze(-1) * expert_logits
        expert_weight = None
        if self.is_group_weight:
            mix_logits = self.group_weight(expert_logits)
        else:
            if self.hparams.is_weight_avg:
                mix_logits = torch.mean(expert_logits, dim=1)
            else:

                mix_logits, expert_weight = self.group_weight(tgt_feature, expert_logits, tgt_domain_onehot)


        loss = self.cross_entropy_loss(mix_logits, tgt_rumor_y)
        loss_first = self.hparams.hyper_beta * loss_first
        # lambda
        # loss_first
        # + loss_first
        if hasattr(self.hparams, "is_only_weak") and self.hparams.is_only_weak:
            loss = loss
        else:
            loss = loss + loss_first
        if expert_weight is None:
            expert_weight = torch.tensor([0.1, 0.2, 0.3])
        return loss, mix_logits, loss, expert_weight

    def training_step(self, batch, batch_idx, optimizer_idx):

        inputs = {"src_rumor_y": batch[0], "src_domain_y": batch[1], "src_x": batch[2],
                  "tgt_rumor_y": batch[3], 'tgt_x': batch[5], 'tgt_domain_y': batch[4],
                  "tgt_no_rumor_y": batch[6], 'tgt_no_x': batch[8], 'tgt_no_domain_y': batch[7],
                  }
        (opt_expert, opt_weight) = self.optimizers()
        if self.current_epoch < self.hparams.pre_train_epochs:
            # first stage
            loss = self.first_stage(src_x=inputs['src_x'], src_domain_y=inputs['src_domain_y'],
                             src_rumor_y=inputs['src_rumor_y'], tgt_x=inputs['tgt_no_x'],
                             tgt_domain_y=inputs['tgt_no_domain_y'])[0]
            self.manual_backward(loss, opt_expert)
            opt_expert.step()
            opt_expert.zero_grad()
        else:

            loss = self.second_stage(src_x=inputs['src_x'], src_domain_y=inputs['src_domain_y'],
                             src_rumor_y=inputs['src_rumor_y'], tgt_x=inputs['tgt_x'], tgt_rumor_y=inputs['tgt_rumor_y'],
                             tgt_domain_y=inputs['tgt_domain_y'], tgt_no_x=inputs['tgt_no_x'], tgt_no_domain_y=inputs['tgt_no_domain_y'],
                                     tgt_no_rumor_y=inputs['tgt_no_rumor_y']
                                     )[0]
            self.manual_backward(loss, opt_weight)
            opt_weight.step()
            opt_weight.zero_grad()
        # instance_weight = self.group_weight.weight.data
        tensorboard_logs = {"loss": loss}
        # return {"loss": loss, "log": tensorboard_logs,"instance_weight": instance_weight}
        return {"loss": loss, "log": tensorboard_logs}


    def training_epoch_end(self, outputs):
        loss = np.mean([i['loss'].item() for i in outputs[0]])
        # loss_s = np.mean([i['log']['loss_s'].item() for i in outputs])
        # loss_g = np.mean([i['log']['loss_g'].item() for i in outputs])
        # instance_weight_list = np.concatenate([i['instance_weight'].detach().cpu().numpy() for i in outputs[0]])
        self.logger.experiment.add_scalar("Train/" + "Loss",
                                          loss, self.current_epoch)
        # self.logger.experiment.add_scalar("Train/" + "Loss_Sources",
        #                                   loss_s, self.current_epoch)
        # self.logger.experiment.add_scalar("Train/" + "Loss_Target",
        #                                   loss_g, self.current_epoch)
        # self.logger.experiment.add_histogram("Train/InstanceWeight", instance_weight_list, self.current_epoch)


    def validation_step(self, batch, batch_idx):
        inputs = {"src_rumor_y": batch[0], "src_domain_y":batch[1], "src_x": batch[2],
                  'tgt_rumor_y':batch[3], 'tgt_domain_y':batch[4],'tgt_x':batch[5]}
        g_input = {"y": batch[3], 'x': batch[5], "domain_y": batch[4]}
        if self.current_epoch < self.hparams.pre_train_epochs:
            # first stage
            output = self.first_stage(src_x=inputs['src_x'], src_domain_y=inputs['src_domain_y'],
                                    src_rumor_y=inputs['src_rumor_y'], tgt_x=inputs['tgt_x'],
                                    tgt_domain_y=inputs['tgt_domain_y'])
            loss = output[1][0]
            src_logits = output[1][1]
            src_logits = torch.softmax(src_logits, dim=-1).detach().cpu().numpy()
            src_labels = batch[0].detach().cpu().numpy()
            return {"val_loss": loss.detach().cpu(), "tgt_loss": loss,
                    "logits": src_logits, "target": src_labels}

        else:
            # loss + loss_first, mix_logits, loss
            loss, mix_logits, clf_loss, _ = self.second_stage(src_x=inputs['src_x'], src_domain_y=inputs['src_domain_y'],
                                     src_rumor_y=inputs['src_rumor_y'], tgt_x=inputs['tgt_x'], tgt_rumor_y=inputs['tgt_rumor_y'],
                                     tgt_domain_y=inputs['tgt_domain_y'])

            tgt_logits = torch.softmax(mix_logits, dim=-1).detach().cpu().numpy()
            tgt_labels = batch[3].detach().cpu().numpy()

            return {"val_loss": clf_loss.detach().cpu(), "tgt_loss": loss,
                    "logits": tgt_logits, "target": tgt_labels}


    def test_step(self, batch, batch_idx):
        inputs = {'tgt_x': batch[5], 'tgt_domain_y': batch[4],"tgt_rumor_y": batch[3]}
        _, mix_logits, _, expert_weight = self.second_stage(src_x=inputs['tgt_x'], src_domain_y=inputs['tgt_domain_y'],
                                             src_rumor_y=inputs['tgt_rumor_y'], tgt_x=inputs['tgt_x'],
                                             tgt_domain_y=inputs['tgt_domain_y'], tgt_rumor_y=inputs['tgt_rumor_y'])

        tgt_logits = torch.softmax(mix_logits, dim=-1).detach().cpu().numpy()
        tgt_labels = batch[3].detach().cpu().numpy()
        expert_weight = expert_weight.detach().cpu().numpy()
        # g_input = {"y": batch[3], 'x': batch[5], "domain_y": batch[4]}
        # tgt_outputs = self(**g_input)
        loss = -1

        return {"test_loss": loss,
                "logits": tgt_logits, "target": tgt_labels, "expert_weight":expert_weight}

    def test_eval_end(self, outputs) -> tuple:
        # val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        try:
            loss = np.mean([[i[j].cpu().item() for j in i.keys() if "loss" in j] for i in outputs])
        except:
            loss = -1
        logits = np.concatenate([x["logits"] for x in outputs], axis=0)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        expert_weight = np.concatenate([x["expert_weight"] for x in outputs], axis=0)

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {"loss": loss, **evaluation(logits, out_label_ids)}
        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list, expert_weight, logits

    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets, expert_weight,logits = self.test_eval_end(outputs)
        logs = ret["log"]
        for key, value in logs.items():
            # self.logger.experiment.add_scalar("Test/" + key + "_s:{}-t:{}".format(self.hparams.src_domain, self.hparams.tgt_domain),
            self.logger.experiment.add_scalar("Test/" + key,
                                              value, self.current_epoch)
        torch.save(expert_weight, self.logger.experiment.log_dir+"/weight.torch")
        torch.save(logits, self.logger.experiment.log_dir+"/logits.torch")
        return {"avg_test_loss": 0, "log": logs}


    def get_loader(self, type,flag=True):
        batch_size = self.hparams.train_batch_size
        selected_dataset = self.get_dataset(type, flag)
        dataloader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size)
        return dataloader


    def on_train_start(self):
        pl.seed_everything(self.hparams.random_seed)

    def get_dataset(self, type, flag):
        if self.hparams.dataset == "text":
            if self.hparams.is_eann:
                dataset = EANNTextDataset
            else:
                dataset = AdvTextDataset
        elif self.hparams.dataset == "comment":
            dataset = CommenetDataset
        else:
            raise NotImplementedError


        selected_dataset = dataset(self.hparams, type, weak_flag=flag, tokenizer=None)
        return selected_dataset

    def configure_optimizers(self):
        model = self.TextClf.model
        domain_adv = self.domain_adv
        group_weight = self.group_weight
        classifier = self.classifiers
        main_optimizer = \
            torch.optim.Adam(filter(lambda p: p.requires_grad, chain(model.parameters(),
                                                                     domain_adv.parameters(), classifier.parameters())),
                             lr=self.hparams.lr_rate
                             )

        group_optimizer = torch.optim.Adam(
            [{"params": filter(lambda p: p.requires_grad, chain(model.parameters(),
                                                    domain_adv.parameters(), classifier.parameters())),
              "lr":self.hparams.main_lr_rate, "weight_decay": self.hparams.weight_decay
              },
             {"params": filter(lambda p: p.requires_grad, group_weight.parameters())}
             ],

            lr=self.hparams.group_lr_rate
        )

        return [main_optimizer, group_optimizer]

    def train_dataloader(self):
        flag = self.current_epoch < self.hparams.pre_train_epochs
        dataloader = self.get_loader(type="train", flag=flag)
        return dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--domain_layer1', type=int, default=64)
        parser.add_argument('--domain_layer2', type=int, default=64)
        parser.add_argument('--domain_class', type=int, default=3)
        parser.add_argument('--lambda',  type=float, default=0.5)
        parser.add_argument('--hyper_beta',  type=float, default=0.5)
        parser.add_argument('--lr_rate',  type=float, default=1e-3)
        parser.add_argument('--group_lr_rate',  type=float, default=1e-3)
        parser.add_argument('--is_eann',  action="store_true")
        # parser.add_argument('--is_flip_label',  action="store_true")
        parser.add_argument('--is_group_weight',  action="store_true")
        parser.add_argument('--weak_labels_path',  type=str, default="./data/top/weak_label_all.csv")
        parser.add_argument('--cls_emb',  type=int, default=128)
        parser.add_argument('--gw_hidden_dim',  type=int, default=64)


        # model structure setting
        parser.add_argument('--is_omit_domain_weight', action="store_true")
        parser.add_argument('--is_sigmoid_weight', action="store_true")
        parser.add_argument('--is_omit_logits', action="store_true")
        parser.add_argument('--is_weight_avg', action="store_true")
        parser.add_argument('--is_only_weak', action="store_true")
        parser.add_argument('--is_softmax_weight', action="store_true")
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--main_lr_rate', type=float, default=0.001)
        parser.add_argument("--weak_fn", default="swear", type=str)
        parser.add_argument("--weak_label_count", default=30, type=int)
        parser.add_argument("--is_debug", action="store_true")


        return parser

