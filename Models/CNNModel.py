import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Text(nn.Module):

    def __init__(self, args, embedding_weight=None):
        super(CNN_Text, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = list(map(int, args.kernel_sizes.split(",")))

        if embedding_weight is not None:
            V = embedding_weight.shape[0]
            D = embedding_weight.shape[1]
            self.embed = nn.Embedding(V, D).from_pretrained(embedding_weight)
        else:
            self.embed = nn.Embedding(V, D)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.cnn_drop_out)


        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc2 = nn.Linear(len(Ks) * Co, self.args.hidden_size)
        self.loss = nn.CrossEntropyLoss()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, y=None, **kwargs):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        hidden = self.fc2(x)
        logit = self.fc1(x)  # (N, C)
        output = (hidden, logit,)
        if y is not None:
            loss = self.loss(logit, y)
            output += (loss,)
        return output
