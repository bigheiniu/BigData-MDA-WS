import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
import torch.nn as nn
from Models import LSTMAttn_Text, CNN_Text, Defend_Text
import pandas as pd
from Util import data_split, evaluation
import numpy as np
from Dataset import TextDataset, CommenetDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


class TextClf(pl.LightningModule):
    def __init__(self, hparams, model_name):
        super(TextClf, self).__init__()
        self.hparams = hparams
        self.step_count = 0
        roberta_config = AutoConfig.from_pretrained("xlm-roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        roberta_model = AutoModel.from_pretrained("xlm-roberta-base", config=roberta_config)
        embedding_weight = roberta_model.get_input_embeddings().weight
        del roberta_model
        self.model_name = model_name
        if model_name == "cnn":
            model = CNN_Text(hparams, embedding_weight=embedding_weight)
        elif model_name == "lstm":
            model = LSTMAttn_Text(hparams, embedding_weight=embedding_weight)
        elif model_name == "defend":
            model = Defend_Text(hparams, embedding_weight=embedding_weight)
        else:
            raise NotImplementedError

        self.model = model

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        if self.model_name == "defend":
            inputs = {"y": batch[0], "news_tokens":batch[2], "comments_tokens": batch[1]}
        else:
            inputs = {"y": batch[0], "x": batch[1]}
        outputs = self(**inputs)
        loss = outputs[1]
        # eval_metrics = outputs[-1]
        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        if self.model_name == "defend":
            inputs = {"y": batch[0], "news_tokens": batch[2], "comments_tokens": batch[1]}
        else:
            inputs = {"y": batch[0], "x": batch[1]}
        outputs = self(**inputs)
        logits = outputs[0]
        loss = outputs[1]
        labels = inputs['y'].detach().cpu().numpy()
        logits = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        return {"val_loss": loss.detach().cpu(), "logits": logits, "target": labels}


    def _eval_end(self, outputs) -> tuple:
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu().item()
        logits = np.concatenate([x["logits"] for x in outputs], axis=0)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {**{"val_loss": val_loss_mean}, **evaluation(logits, out_label_ids)}
        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs}

    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"avg_test_loss": logs["val_loss"], "log": logs}

    def configure_optimizers(self):
        model = self.model
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)


    def get_loader(self, type):
        if self.hparams.dataset == "text":
            dataset = TextDataset
        elif self.hparams.dataset == "comment":
            dataset = CommenetDataset
        else:
            raise NotImplementedError

        batch_size = self.hparams.train_batch_size
        train_dataset = dataset(self.hparams, type, self.tokenizer)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        return dataloader

    def train_dataloader(self):
        dataloader = self.get_loader(type="train")
        return dataloader

    def val_dataloader(self):
        dataloader = self.get_loader(type="val")
        return dataloader

    def test_dataloader(self):
        dataloader = self.get_loader(type="val")
        return dataloader

