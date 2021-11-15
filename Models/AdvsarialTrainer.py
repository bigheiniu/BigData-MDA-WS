import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
import torch.nn as nn
from Models import LSTMAttn_Text, CNN_Text, Defend_Text
import pandas as pd
from Util import data_split, evaluation
import numpy as np
from Dataset import TextDataset, CommenetDataset
from Dataset import AdvTextDataset, EANNTextDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaForSequenceClassification
from Models.TextTrainer import TextClf
from Models.SimpleTrainer import SimpleTrainer
from Models.DomainClf import DomainClf
from Models.DistributionDistance import LabelDD
from itertools import chain
from Models.Layers import ReverseLayerF
from argparse import ArgumentParser
import torch.nn.functional as F

class AdTextClf(SimpleTrainer):
    def __init__(self, hparams, textClf_ckpt):
        super(AdTextClf, self).__init__(hparams)
        self.domain_adv = DomainClf(hparams)
        self.is_cssa = hparams.is_cssa
        self.pre_train_epochs = hparams.pre_train_epochs
        if textClf_ckpt is not None:
            self.TextClf = TextClf.load_from_checkpoint(textClf_ckpt)
        else:
            self.TextClf = TextClf(hparams, model_name='cnn')
        self.hyper_lambda = hparams.hyper_lambda
        self.hparams = hparams
        if hasattr(self.hparams, "is_weak_label") is False:
            self.label_weak = False
        elif self.hparams.is_weak_label:
            self.label_weak = True
        else:
            self.label_weak = False

        if self.label_weak:
            self.label_kl = LabelDD(distance_metric='kl')

        # self.reverse_layer = ReverseLayerF()



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


    def csa_loss(self, src_feature, tgt_feature, class_eq):
        margin = 1
        dist = F.pairwise_distance(src_feature, tgt_feature)
        loss = class_eq * dist.pow(2)
        loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
        return loss.mean()




    def forward(self, src_x=None, src_domain_y=None, src_rumor_y=None, tgt_x=None, tgt_domain_y=None, tgt_rumor_y=None):
        tgt_features, tgt_logits = self.TextClf.forward(x=tgt_x)
        outputs = (tgt_logits,)
        if src_x is not None:
            src_features, src_logits, src_rumor_loss = self.TextClf.forward(x=src_x, y=src_rumor_y)
            src_reverse_features = ReverseLayerF.apply(src_features, self.hyper_lambda)
            tgt_reverse_features = ReverseLayerF.apply(tgt_features, self.hyper_lambda)
            _, src_domain_loss = self.domain_adv(src_reverse_features, src_domain_y)
            _, tgt_domain_loss = self.domain_adv(tgt_reverse_features, tgt_domain_y)
            # The domain loss already times the domain lambda in the reverse layer
            loss = 0
            if self.current_epoch > self.pre_train_epochs:
                # pre-train the domain classifier
                loss += src_rumor_loss
            if self.label_weak:
                kl_label_loss = self.label_kl(tgt_logits)
                loss += kl_label_loss
            if self.is_cssa:
                #TODO CSSA model
                loss =(1 - self.hyper_lambda) * src_rumor_loss + self.hyper_lambda * \
                        self.csa_loss(src_features, tgt_features, (src_rumor_y == tgt_rumor_y).float())
            else:
                loss += tgt_domain_loss + src_domain_loss

            outputs += (loss, src_rumor_loss, src_domain_loss, tgt_domain_loss,)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs = {"src_rumor_y": batch[0], "src_domain_y":batch[1], "src_x": batch[2],
                  'tgt_x': batch[5], 'tgt_domain_y':batch[4], "tgt_rumor_y":batch[3]}
        loss = self.forward(**inputs)
        loss = loss[1]
        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # torch.tensor(self.src_labels[item]), torch.tensor(self.src_domain[item]), torch.tensor(self.src_features[item]), \
        #         torch.tensor(self.tgt_labels[item]), torch.tensor(self.tgt_domain[item]), torch.tensor(self.tgt_features[item])
        inputs = {"src_rumor_y": batch[0], "src_domain_y":batch[1], "src_x": batch[2],
                  'tgt_domain_y':batch[4],'tgt_x':batch[5], 'tgt_rumor_y':batch[3]}
        outputs = self(**inputs)
        tgt_logits = outputs[0]
        loss = outputs[1]
        src_rumor_loss = outputs[2]
        src_domain_loss = outputs[3]
        tgt_domain_loss = outputs[4]
        labels = batch[3].cpu().numpy()
        tgt_logits = torch.softmax(tgt_logits, dim=-1).cpu().numpy()

        return {"val_loss": loss.detach().cpu(), "src_rumor_loss": src_rumor_loss,
                "src_domain_loss":src_domain_loss, "tgt_domain_loss":tgt_domain_loss,
                "logits": tgt_logits, "target": labels}


    def test_step(self, batch, batch_idx):
        inputs = {'tgt_x': batch[5], 'tgt_domain_y': batch[4]}
        outputs = self(**inputs)
        loss = -1
        tgt_logits = outputs[0]
        labels = batch[3].detach().cpu().numpy()
        tgt_logits = torch.softmax(tgt_logits, dim=-1).detach().cpu().numpy()

        return {"test_loss": loss,
                "logits": tgt_logits, "target": labels}

    def get_loader(self, type):
        batch_size = self.hparams.train_batch_size
        selected_dataset = self.get_dataset(type)
        dataloader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size)
        return dataloader

    def get_dataset(self, type):
        if self.hparams.dataset == "text":
            if self.hparams.is_eann:
                dataset = EANNTextDataset
            else:
                dataset = AdvTextDataset
        elif self.hparams.dataset == "comment":
            dataset = CommenetDataset
        else:
            raise NotImplementedError


        selected_dataset = dataset(self.hparams, type, tokenizer=None)
        return selected_dataset

    def configure_optimizers(self):
        model = self.TextClf.model
        domain_adv = self.domain_adv
        main_optimizer = \
            torch.optim.Adam(filter(lambda p: p.requires_grad, chain(model.parameters(), domain_adv.parameters())),
                             lr=self.hparams.lr_rate
                             )

        return [main_optimizer]


    def setup(self, stage):
        if self.label_weak:
            if stage == "fit":
                p_y = self.get_dataset('train').p_y
                p_y = p_y.to(self.device)
                self.label_kl.set_p_y(p_y)



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--domain_layer1', type=int, default=64)
        parser.add_argument('--domain_layer2', type=int, default=64)
        # 100 for eann
        parser.add_argument('--domain_class', type=int, default=3)
        parser.add_argument('--lambda',  type=float, default=0.5)
        parser.add_argument('--lr_rate',  type=float, default=1e-3)
        parser.add_argument('--is_eann',  action="store_true")
        parser.add_argument('--is_weak_label',  action="store_true")
        return parser