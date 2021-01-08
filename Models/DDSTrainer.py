import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
import torch.nn as nn
from Models import LSTMAttn_Text, CNN_Text, Defend_Text
import pandas as pd
from Util import data_split, evaluation
import numpy as np
from Dataset import TextDataset, CommenetDataset
from Dataset import AdvTextDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaForSequenceClassification
from Models.TextTrainer import TextClf
from Models.SimpleTrainer import SimpleTrainer
from Models.DomainClf import DomainClf
from itertools import chain
from Models.Layers import ReverseLayerF
from argparse import ArgumentParser
from torch.optim import Optimizer
from typing import Optional



class DDSTextClf(SimpleTrainer):
    def __init__(self, hparams):
        super(DDSTextClf, self).__init__(hparams)
        self.domain_adv = DomainClf(hparams)
        self.TextClf = TextClf(hparams)
        self.hparams = hparams


    def forward(self, src_x=None, src_rumor_y=None, **kwargs):
        outputs = self.TextClf.forward(x=src_x, y=src_rumor_y)
        return outputs

    def first_phase_forward(self, src_x, src_domain_y, tgt_x, tgt_domain_y, **kwargs):
        src_features = self.TextClf.forward(x=src_x)[0]
        tgt_features = self.TextClf.forward(x=tgt_x)[0]
        loss_src_domain = self.domain_adv(src_features, src_domain_y)[1]
        loss_tgt_domain = self.domain_adv(tgt_features, tgt_domain_y)[1]
        loss = loss_src_domain + loss_tgt_domain
        return loss

    def second_phase_forward(self, src_x, src_rumor_y, tgt_domain_y, **kwargs):
        src_features = self.TextClf.forward(x=src_x)[0]
        tgt_domain_index = tgt_domain_y.reshape(-1)[0]
        sorted_index = self.domain_adv.curriculum_learning(src_features, tgt_domain_index)
        # train with the most relevant samples
        batch_size = len(src_x)
        src_x = src_x[sorted_index[:int(self.hparams.sample_ratio * batch_size)]]
        src_rumor_y = src_rumor_y[sorted_index[:int(self.hparams.sample_ratio * batch_size)]]
        return self.forward(src_x, src_rumor_y)[-1]

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = {"src_rumor_y": batch[0], "src_domain_y":batch[1], "src_x": batch[2],
                  'tgt_x': batch[5], 'tgt_domain_y':batch[4]}
        if self.current_epoch < self.hparams.pre_train_epochs:
            # first phase
            loss = self.first_phase_forward(**inputs)
        else:
            # second and third phase
            loss = self.second_phase_forward(**inputs)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs = {"src_rumor_y": batch[0], "src_domain_y":batch[1], "src_x": batch[2],
                  'tgt_x':batch[5], 'tgt_domain_y':batch[4]}
        outputs = self(**inputs)
        tgt_logits = outputs[1]
        loss = outputs[-1]
        labels = batch[3].detach().cpu().numpy()
        tgt_logits = torch.softmax(tgt_logits, dim=-1).detach().cpu().numpy()

        return {"val_loss": loss.detach().cpu(),
                "logits": tgt_logits, "target": labels}


    def test_step(self, batch, batch_idx):
        inputs = {'tgt_x': batch[5], 'tgt_domain_y': batch[4]}
        outputs = self(inputs['tgt_x'], None)
        loss = -1
        tgt_logits = outputs[1]
        labels = batch[3].detach().cpu().numpy()
        tgt_logits = torch.softmax(tgt_logits, dim=-1).detach().cpu().numpy()

        return {"test_loss": loss,
                "logits": tgt_logits, "target": labels}

    def get_loader(self, type):
        if self.hparams.dataset == "text":
            dataset = AdvTextDataset
        elif self.hparams.dataset == "comment":
            dataset = CommenetDataset
        else:
            raise NotImplementedError

        batch_size = self.hparams.train_batch_size
        selected_dataset = dataset(self.hparams, type, tokenizer=None)
        dataloader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size)
        return dataloader


    def configure_optimizers(self):
        model = self.TextClf.model
        domain_adv = self.domain_adv
        first_phase_optimizer = \
            torch.optim.Adam(filter(lambda p: p.requires_grad, chain(model.parameters(), domain_adv.parameters())),
                             lr=self.hparams.first_lr_rate
                             )
        second_phase_optimizer = \
            torch.optim.Adam(filter(lambda p: p.requires_grad, chain(model.parameters(), domain_adv.parameters())),
                             lr=self.hparams.second_lr_rate
                             )

        return [first_phase_optimizer, second_phase_optimizer]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        second_order_closure,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        if optimizer_idx == 0 and epoch < self.hparams.pre_train_epochs:
            # first phase optimization
            optimizer.step()
            optimizer.zero_grad()

        elif optimizer_idx == 1 and epoch >= self.hparams.pre_train_epochs:
            # second phase optimization
            optimizer.step()
            optimizer.zero_grad()


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--domain_layer1', type=int, default=64)
        parser.add_argument('--domain_layer2', type=int, default=64)
        parser.add_argument('--domain_class', type=int, default=3)
        parser.add_argument('--first_lr_rate',  type=float, default=1e-3)
        parser.add_argument('--second_lr_rate',  type=float, default=1e-4)
        parser.add_argument('--pre_train_epochs',  type=int, default=15)
        parser.add_argument('--sample_ratio',  type=float, default=0.2)

        return parser