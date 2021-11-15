import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback


class LabelDD(nn.Module):
    def __init__(self, distance_metric="KL"):
        super(LabelDD, self).__init__()
        if distance_metric == "kl":
            self.loss_fn = torch.nn.KLDivLoss(size_average=False)
        else:
            raise NotImplementedError
        self.dummy_param = nn.Parameter(torch.empty(0))

    def set_p_y(self, p_y):
        self.p_y = p_y

    def label_probability(self, y_true_all):
        _, _, counts = torch.unique(y_true_all, sorted=True, return_counts=True)
        label_pro = counts.float() / torch.sum(counts)
        return label_pro

    def pred_probability(self, pred_logits):
        batch_size = float(pred_logits.shape[0])
        p_y_batch = torch.sum(torch.softmax(pred_logits, dim=1), dim=0) / batch_size
        return p_y_batch

    def forward(self, pred_logits, p_y=None):
        p_y_pred = self.pred_probability(pred_logits)
        p_y = p_y if p_y is not None else self.p_y
        p_y = p_y.to(p_y_pred.device)
        return self.loss_fn(p_y, p_y_pred)





