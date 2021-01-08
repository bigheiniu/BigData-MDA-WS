import torch
import torch.nn as nn
from Models.DomainClf import MMD_loss

class DomainAdaptation(nn.Module):
    def __init__(self, args, base_model):
        super(DomainAdaptation, self).__init__()
        self.base_model = base_model
        self.kernel_mul = args.kernel_mul
        self.kernel_num = args.kernel_num
        self.mmd_loss = MMD_loss(self.kernel_mul, self.kernel_num)
        self.hyper_lambda = args.hyper_lambda

    def forward(self, source_x, source_y, target_x):
        source_feature, _, clf_loss = self.base_model(source_x, source_y)
        tgt_feature, _, clf_loss = self.base_model(target_x)
        domain_loss = self.mmd_loss(source_feature, tgt_feature)
        loss = clf_loss + self.hyper_lambda * domain_loss
        return loss




