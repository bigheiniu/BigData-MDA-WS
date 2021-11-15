import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
from Util import data_split, evaluation, set_random_seed
from Models.AdvsarialTrainer import AdTextClf
from Models.DDSTrainer import DDSTextClf
from Models.MetaTrainer import MetaClf
# from run_in_one import build_args
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import os
from glob import glob
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np


# RoBERTaClf XLM
# RoBERTa model on English and another on disinformation.

import shutil
def activate_model_init(hparams):

    if hparams.model_type == 'adv':

        try:
            # checkpoint_path = glob(
            # f"tb_logs/cnns:{hparams.src_domain}-t:{hparams.tgt_domain}/version_0/checkpoints/*.ckpt")[
            # 0]
            checkpoint_path = None
        except:
            checkpoint_path = None
        model = AdTextClf(hparams, textClf_ckpt=checkpoint_path)
    elif hparams.model_type == 'new':
        model = MetaClf(hparams)
    else:
        model = DDSTextClf(hparams)
    return model

def nn_clf_method(method_name, parser):
    # set the random seed for each experiment
    # parser = AdTextClf.add_model_specific_args(parser)
    parser = MetaClf.add_model_specific_args(parser)
    hparams = parser.parse_args()
    print(hparams)
    if hparams.tgt_domain in hparams.src_domain:
        print("TGT domain in the source domain!")
        exit()
    model = activate_model_init(hparams)
    tag = "_eann_" if hparams.is_eann else ""
    tag += "_not_" if hparams.is_not_in else ""
    tag += "_cssa_" if hparams.is_cssa else ""
    # tag += "_ws_" if hparams.is_weak_label else ""
    tag += "_flip" if hparams.is_flip_label else ""
    tag += hparams.special_tag
    if hparams.hyper_lambda == 0.0:
        model_type = "CNN"

    else:
        model_type = hparams.model_type

    logger = TensorBoardLogger('tb_logs', name=model_type + tag + method_name + tag + "s:{}-t:{}-random:{}".format(hparams.src_domain, hparams.tgt_domain,
            hparams.random_seed))
    flag = hparams.model_type != "new"
    pl.seed_everything(hparams.random_seed)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        save_top_k=1,
        mode='max')
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        logger=logger,
        # reload_dataloaders_every_epoch=True,
        callbacks=[checkpoint_callback],
        automatic_optimization=flag
        # checkpoint_callback=ModelCheckpoint(monitor='val_loss', mode='min', filepath=logger.log_dir+"/checkpoints/{epoch:02d}-{val_loss:.2f}")
    )
    trainer.fit(model)
    # utilize the last model
    trainer.test(model)
    # utilize the best model
    trainer.test(ckpt_path=checkpoint_callback.best_model_path)
    # shutil.rmtree(logger.log_dir + "/checkpoints")
    # for file in glob(""
    #                  + 'tb_logs/'
    #                  + hparams.model_type + "_" + method_name + tag + "s:{}-t:{}".format(hparams.src_domain,
    #                                                                                      hparams.tgt_domain)
    #                  + "/**/checkpoints"):
    #     shutil.rmtree(file)


def evaluate(method_name, parser):
    parser = MetaClf.add_model_specific_args(parser)
    hparams = parser.parse_args()
    tag = "_eann_" if hparams.is_eann else ""

    # checkpoint_path = glob(f"tb_logs/{method_name}s:{hparams.src_domain}-t:{hparams.src_domain}/version_0/checkpoints/*.ckpt")[0]
    checkpoint_path = "/home/yli29/FakeDetectionBaseline/tb_logs/newweight_analysiscnnweight_analysiss:politi,health_deterrent-t:gossip/version_27/checkpoints/epoch=23.ckpt"
    logger = TensorBoardLogger('tb_logs',
                               name=method_name + tag + "s:{}-t:{}".format(hparams.src_domain, hparams.tgt_domain))
    model = activate_model_init(hparams)
    model = model.load_from_checkpoint(checkpoint_path,  hparams=hparams, model_name=method_name,textClf_ckpt=None)
    model.hparams = hparams
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        logger=logger,
        checkpoint_callback=False
    )
    trainer.test(model)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/all_data.csv")
    parser.add_argument("--clf_method", type=str, default="cnn", choices=['logistic', 'svm', 'xgb', 'roberta', 'lstm', 'cnn', 'defend'])
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--max_comment_length", type=int, default=50)
    parser.add_argument("--max_sentence_length", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="text")
    parser.add_argument("--sentence_count", type=int, default=10)
    parser.add_argument("--comment_count", type=int, default=10)
    parser.add_argument("--src_domain", default="gossip", type=str)
    parser.add_argument("--tgt_domain", type=str, default="gossip")
    parser.add_argument("--is_few_shot", action="store_true")
    parser.add_argument("--tgt_train_size", default=0.1)
    parser.add_argument("--is_not_in", action="store_true")
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--random_seed", default=123, type=int)
    parser.add_argument("--model_type", default="adv", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--is_flip_label", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.4, help="flip the label of the samples with probability xx")

    nn_model_args = parser.add_argument_group("neural network ckpt parameters")
    nn_model_args.add_argument("--embed_num", default=-1, type=int)
    nn_model_args.add_argument("--embed_dim", default=-1, type=int)
    nn_model_args.add_argument("--class_num", default=2, type=int)
    nn_model_args.add_argument("--kernel_num", default=100, type=int)
    nn_model_args.add_argument("--kernel_sizes", default="3,4,5", type=str)
    nn_model_args.add_argument("--cnn_drop_out", default=0.3, type=float)
    nn_model_args.add_argument("--hidden_size", default=100, type=int)
    nn_model_args.add_argument("--train_batch_size", default=32, type=int)
    nn_model_args.add_argument("--eval_batch_size", default=32, type=int)
    nn_model_args.add_argument("--epochs", default=30, type=int)
    nn_model_args.add_argument("--pre_train_epochs", default=10, type=int)
    nn_model_args.add_argument("--class_count", default=2, type=int)
    nn_model_args.add_argument("--is_weak_label", action="store_true", dest="For DDT Approach")
    nn_model_args.add_argument("--is_get_clean_data", action="store_true")
    nn_model_args.add_argument("--is_cssa", action="store_true")
    nn_model_args.add_argument("--clean_count", default=10, type=int)
    nn_model_args.add_argument("--special_tag", default="", type=str)






    for method_name in ['cnn']:
        nn_clf_method(method_name, parser)
        # evaluate(method_name, parser)




