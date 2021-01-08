import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
import torch.nn as nn
from Models import LSTMAttn_Text, CNN_Text, Defend_Text
import pandas as pd
from Util import data_split, evaluation
import numpy as np
from Dataset import TextDataset, CommenetDataset
from Dataset import SimpleTextDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaForSequenceClassification


class SimpleTrainer(pl.LightningModule):
    def __init__(self, hparams):
        super(SimpleTrainer, self).__init__()
        self.hparams = hparams

    def forward(self, **inputs):
        return self.model(**inputs)

    def _eval_end(self, outputs) -> tuple:
        # val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        try:
            loss = np.mean([[i[j].cpu().item() for j in i.keys() if "loss" in j] for i in outputs])
        except:
            loss = -1
        logits = np.concatenate([x["logits"] for x in outputs], axis=0)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {"loss":loss, **evaluation(logits, out_label_ids)}
        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        for key, value in logs.items():
            # self.logger.experiment.add_scalar("Val/" + key + "_s:{}-t:{}".format(self.hparams.src_domain, self.hparams.tgt_domain),
            self.logger.experiment.add_scalar("Val/" + key,
                                              value, self.current_epoch)
        return {"val_loss": logs['loss'], "log": logs, 'val_acc': logs['acc']}

    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets = self._eval_end(outputs)
        logs = ret["log"]
        for key, value in logs.items():
            # self.logger.experiment.add_scalar("Test/" + key + "_s:{}-t:{}".format(self.hparams.src_domain, self.hparams.tgt_domain),
            self.logger.experiment.add_scalar("Test/" + key ,
                                              value, self.current_epoch)
        return {"avg_test_loss": 0, "log": logs}

    def configure_optimizers(self):
        model = self.model
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        self.opt = optimizer
        return [optimizer]


    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)


    def get_loader(self, type):
        if self.hparams.dataset == "text":
            dataset = SimpleTextDataset
        elif self.hparams.dataset == "comment":
            dataset = CommenetDataset
        else:
            raise NotImplementedError

        batch_size = self.hparams.train_batch_size
        selected_dataset = dataset(self.hparams, type, self.tokenizer)
        dataloader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size)
        return dataloader

    def train_dataloader(self):
        dataloader = self.get_loader(type="train")
        return dataloader

    def val_dataloader(self):
        dataloader = self.get_loader(type="val")
        return dataloader

    def test_dataloader(self):
        dataloader = self.get_loader(type="test")
        return dataloader

