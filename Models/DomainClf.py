import torch
import torch.nn as nn

class DomainClf(nn.Module):
    def __init__(self, hparams):
        super(DomainClf, self).__init__()
        self.hparams1 = hparams
        self.domain_clf = nn.Sequential(
            nn.Linear(self.hparams1.hidden_size, self.hparams1.domain_layer1),
            nn.ReLU(),
            nn.Linear(self.hparams1.domain_layer1, self.hparams1.domain_layer2),
            nn.Linear(self.hparams1.domain_layer2, self.hparams1.domain_class)
        )
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cross_entropy_seperate = nn.CrossEntropyLoss(reduction="none")
        final_dim = (self.hparams1.domain_class - 1) * 2
        self.domain_logit_clf = nn.Sequential(
            nn.Linear(self.hparams1.hidden_size, self.hparams1.domain_layer1),
            nn.ReLU(),
            nn.Linear(self.hparams1.domain_layer1, self.hparams1.domain_layer2),
            nn.Linear(self.hparams1.domain_layer2, final_dim)
        )

    def set_source(self, target_class):
        self.target_class = target_class
        self.element = [0, 1, 2]
        self.element.pop(target_class)


    def forward(self, hidden, domain_class=None, **kwargs):
        predict_domain = self.domain_clf(hidden)
        output = (predict_domain,)
        if domain_class is not None:
            loss = self.cross_entropy(predict_domain, domain_class)
            output += (loss,)
        return output

    def forward_final(self, hidden, domain_class=None, **kwargs):
        predict_domain = self.domain_logit_clf(hidden)
        output = (predict_domain,)
        if domain_class is not None:
            loss = 0

            for index, source in enumerate(self.element):
                predict_domain_here = predict_domain.reshape(-1, 2, 2)
                if len(torch.count_nonzero(domain_class == source)) > 0:
                    mask = torch.where(domain_class == source, hidden.new_ones(domain_class.shape[0]),
                                       hidden.new_zeros(domain_class.shape[0]))
                    # batch * 1 * 1
                    large_mask = mask.unsqueeze(1).unsqueeze(1)
                    # batch * 1 * 2
                    large_mask = large_mask.repeat(1, 1, 2)
                    other_mask = hidden.new_zeros(domain_class.shape[0], 1, 2)
                    if index == 0:
                        mask = torch.cat([large_mask, other_mask], dim=1)
                    else:
                        mask = torch.cat([other_mask, large_mask], dim=1)
                    predict_pro = torch.sum(mask * predict_domain, dim=1)
                    labels = torch.ones_like(domain_class)
                    loss_all = self.cross_entropy_seperate(predict_pro, labels)
                    loss += torch.mean(mask * loss_all)

            if len(torch.count_nonzero(domain_class == self.target_class)) > 0:
                predict_domain_here = predict_domain.reshape(predict_domain.shape[0] * 2, 2)
                labels = torch.zeros_like(domain_class)
                mask = torch.where(domain_class == self.target_class, hidden.new_ones(domain_class.shape[0]),
                                   hidden.new_zeros(domain_class.shape[0]))
                mask = mask.repeat(2)
                labels = torch.zeros_like(domain_class)
                labels = labels.repeat(2)
                loss_all = self.cross_entropy_seperate(predict_domain_here, labels)
                loss += torch.mean(loss_all * mask)


            loss = self.cross_entropy(predict_domain, domain_class)
            output += (loss,)
        return output

    def curriculum_learning(self, hidden, domain_want):
        logit = self.forward(hidden)[0]
        logit_want = logit[:, domain_want]
        sample_index = torch.argsort(logit_want, descending=True)
        return sample_index

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target, **kwargs):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

