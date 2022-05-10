import torch
import pandas as pd
from Util import data_split, data_split_val
import spacy
import numpy as np
import pickle
# nlp = spacy.load('en_core_web_sm')
import json
import dateparser
# from bson.int64 import Int64
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.linalg import svds
import os
import nltk
from Util import data_preprocess
import json
from transformers import RobertaTokenizer
domain_map = {"gossip":0,"politi":1, "health_deterrent":2}

def flip_label(labels, p=0.4):
    mask = np.random.binomial(1, p, len(labels))
    flip_label = []
    for index, i in enumerate(mask):
        # keep the label
        if i == 0:
            flip_label.append(labels[index])
        else:
            flip_label.append(1-labels[index])
    return flip_label

class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, type, is_tgt, tokenizer):
        self.hparams = hparams
        if hparams.clf_method == 'defend':
            file_name = "./data/simple_text_defend.torch"
        else:
            file_name = "./data/simple_text.torch"
        # file_name = "/home/yli29/FakeDetectionBaseline/data/simple_text.torch"
        if os.path.exists(file_name):
            data = torch.load(file_name)
            for key, value in data.items():
                setattr(self, key, json.loads(value))
        else:
            data = pd.read_csv(hparams.data_path)
            if hparams.clf_method == 'defend':
                news_content_list = data['content'].tolist()
                news_sentences = [[j.text for j in nlp(i).sents] for i in news_content_list]
                # news_sentences = [[j for j in i if len(j) > 10] for i in news_sentences]
                # truncate the senteces
                sentence_count = hparams.sentence_count
                news_sentences = [i[:sentence_count] + (sentence_count - len(i))
                                  * ["<pad>"] for i in news_sentences]
                data_sentences = [[tokenizer.encode(j, max_length=hparams.max_sentence_length,
                                                    truncation=True,
                                                    pad_to_max_length=True
                                                    ) for j in i] for i in news_sentences]
                data['encode_text'] = data_sentences
            else:
                data['encode_text'] = data['content'].apply(lambda x: tokenizer.encode(x,
                                                                                       max_length=hparams.max_length,
                                                                                       truncation=True,
                                                                                       pad_to_max_length=True, ))

            self.data = data.to_json(orient="records")
            torch.save({"data":self.data
                        }, file_name)

        # select specific domain dataset
        self.data = [{**self.data[i], "index":i} for i in range(len(self.data))]
        self.random_seed = hparams.random_seed
        src_domain = hparams.src_domain
        tgt_domain = hparams.tgt_domain
        is_not_in = hparams.is_not_in

        train_src_data, train_src_label, _, _, val_src_data, val_src_label = \
            self.get_domain_data(self.data, src_domain, self.random_seed, is_not_in)
        train_tgt_data, train_tgt_label, test_tgt_data, test_tgt_label, val_tgt_data, val_tgt_label = \
            self.get_domain_data(self.data, tgt_domain, random_seed=self.random_seed, is_not_in=False)

        # print("Attention We are using validation data as the training dataset")
        # test_tgt_data = train_src_data
        # test_tgt_label = train_src_label
        # train_src_data = val_src_data
        # train_src_label = val_src_label



        if src_domain != tgt_domain and type == 'train' and hparams.is_few_shot:
            train_data_tgt, train_label_tgt, _, _ = self.get_domain_data(self.data, tgt_domain, random_seed=self.random_seed)
            hparams.tgt_train_size = float(hparams.tgt_train_size)
            max_len = hparams.tgt_train_size if hparams.tgt_train_size > 1 else int(hparams.tgt_train_size * len(train_data_tgt))
            max_len = int(max_len)
            train_src_data += train_data_tgt[:max_len]
            train_src_label += train_label_tgt[:max_len]
        elif src_domain == tgt_domain and type == 'train' and hparams.is_few_shot:
            max_len = hparams.tgt_train_size if hparams.tgt_train_size > 1 else int(
                hparams.tgt_train_size * len(train_src_data))
            max_len = int(max_len)
            train_src_data = train_src_data[:max_len]
            train_src_label = train_src_label[:max_len]


        if type == "train":
            self.features = [i[0] for i in train_src_data]
            self.labels = train_src_label
            return_element, train_tgt_data_notin = self.get_weak_labels(train_tgt_data)
            if hparams.is_flip_label:
                if hparams.is_only_weak:
                    self.labels = return_element['weak_label'].values.tolist()
                    self.features = return_element['encoded_text'].values.tolist()
                else:
                    self.labels += return_element['weak_label'].values.tolist()
                    self.features += return_element['encoded_text'].values.tolist()

            if self.hparams.is_get_clean_data:
                select_data = train_tgt_data_notin.iloc[:self.hparams.clean_count, :]
                new_index = select_data['index_new']
                select_tgt_features = select_data[['encoded_text']].values.tolist()
                select_tgt_labels = [train_tgt_label[i] for i in new_index]
                assert len(select_tgt_labels) == self.hparams.clean_count
                assert len(select_tgt_features) == self.hparams.clean_count
                self.labels += select_tgt_labels
                self.features += [i[0] for i in select_tgt_features]

        elif type == 'test':
            self.features = [i[0] for i in test_tgt_data]
            self.labels = test_tgt_label
        else:
            self.features = [i[0] for i in val_tgt_data]
            self.labels = val_tgt_label

    # def get_weak_labels(self, tgt_train_data):
    #
    #     data = pd.read_csv(self.hparams.weak_labels_path, header=None)
    #     data = data.rename(columns={0: "index", 1: 'weak_label'})
    #     tgt_train_data = pd.DataFrame(tgt_train_data, columns=['encoded_text', 'index'])
    #     tgt_train_data['index_new'] = list(range(len(tgt_train_data)))
    #     tgt_train_data_in = tgt_train_data.join(data.set_index('index'), how='inner', on='index')
    #     tgt_train_data_notin = tgt_train_data[
    #         tgt_train_data['index'].apply(lambda x: x not in set(data['index'].values.tolist()))]
    #     print("There are {} weak samples".format(len(tgt_train_data)))
    #     tgt_train_data_in['weak_label'] = tgt_train_data_in['weak_label'].apply(lambda x: int(x))
    #     tgt_train_data_in = tgt_train_data_in[['encoded_text', 'weak_label']]
    #
    #     return tgt_train_data_in, tgt_train_data_notin

    def get_weak_label_v2(self, tgt_train_data):
        data = pd.read_csv(self.hparams.weak_labels_path, header=None)
        data = data.loc[tgt_train_data['index'],:]
        weak_labels = data[self.hparams.weak_fn].tolist()

        zero_index = set(data.iloc[list(np.argsort(weak_labels)[:self.hparams.weak_label_count]),'index'].tolist())
        one_index = set(data.iloc[list(np.argsort(weak_labels)[-self.hparams.weak_label_count:]),'index'].tolist())
        tgt_train_data['index_new'] = list(range(len(tgt_train_data)))
        def helper_fn(x):
            if x in zero_index:
                return 0
            elif x in one_index:
                return 1
            else:
                return np.nan
        tgt_train_data['weak_label'] = tgt_train_data['index'].apply(lambda x: helper_fn(x))
        tgt_train_data_notin = tgt_train_data[tgt_train_data['weak_label'].isna()]
        tgt_train_data_in = tgt_train_data[tgt_train_data['weak_label'].notna()]
        tgt_train_data_in = tgt_train_data_in[['encoded_text', 'weak_label', "domain"]]

        return tgt_train_data_in, tgt_train_data_notin

    def get_domain_data(self, self_data, domain, random_seed, is_not_in=False):
        data = []
        if is_not_in is False and "," not in domain:
            for i in self_data:
                if domain in i['domain']:
                    data.append(i)

            # balance the dataset
            one = [(i['encode_text'], i['index']) for i in data if i['label'] == 1]
            zero = [(i['encode_text'], i['index']) for i in data if i['label'] == 0]
            min_len = min(len(one), len(zero))
            one = one[:min_len]
            zero = zero[:min_len]
            data = one + zero
            # train_X, train_Y, test_X, test_Y, val_X, val_Y
            train_data, train_label, test_data, test_label, val_data, val_label = data_split_val(data, [1] * min_len + [
                0] * min_len)
            return train_data, train_label, test_data, test_label, val_data, val_label
        else:
            train_data_list = []
            train_label_list = []
            test_data_list = []
            test_label_list = []
            val_data_list = []
            val_label_list = []
            if is_not_in:
                domains = list(domain_map.keys())
                domains = [i for i in domains if domain not in i]
            else:
                domains = domain.split(",")

            for domain in domains:
                train_data, train_label, test_data, test_label, val_data, val_label = self.get_domain_data(self_data,
                                                                                                      domain,
                                                                                                      random_seed=random_seed)
                print("Domain: {} Train Size: {} Test Size {}".format(domain, len(train_data), len(test_data)))
                train_data_list += train_data
                train_label_list += train_label
                test_data_list += test_data
                test_label_list += test_label
                val_data_list += val_data
                val_label_list += val_label
            return train_data_list, train_label_list, test_data_list, test_label_list, val_data_list, val_label_list
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return torch.tensor(self.labels[item], dtype=torch.long), torch.tensor(self.features[item], dtype=torch.long)

class AdvTextDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, type, tokenizer, weak_flag=False):
        super(AdvTextDataset, self).__init__()
        self.hparams = hparams
        file_name = "./data/simple_text_roberta.torch"
        if os.path.exists(file_name):
            data = torch.load(file_name)
            for key, value in data.items():
                setattr(self, key, json.loads(value))
        else:
            if tokenizer is None:
                tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            data = pd.read_csv(hparams.data_path)
            data['encode_text'] = data['content'].apply(lambda x: tokenizer.encode(x,
                                                                                   max_length=hparams.max_length,
                                                                                   truncation=True,
                                                                                   pad_to_max_length=True, ))
            data['index'] = data.index
            data = data.to_json(orient="records")
            torch.save({"data": data}, file_name)
            self.data = json.loads(data)
        self.random_seed = hparams.random_seed
        # select specific domain dataset
        src_domain = hparams.src_domain
        if "," in src_domain:
            src_domain = src_domain.split(",")
        tgt_domain = hparams.tgt_domain
        is_not_in = hparams.is_not_in

        train_src_data, train_src_label, test_src_data, test_src_label, val_src_data, val_src_label = self.get_domain_data(self.data, src_domain,
                                                                                                self.random_seed, is_not_in)

        train_tgt_data, train_tgt_label, test_tgt_data, test_tgt_label, val_tgt_data, val_tgt_label = self.get_domain_data(self.data,
                                                                                                    tgt_domain, random_seed=self.random_seed)


        self.p_y = self.label_probability(train_tgt_label=torch.tensor(train_tgt_label))

        debug = False

        if type == "train":
            src_features = [(i[0], i[2]) for i in train_src_data]
            src_labels = train_src_label
            tgt_features = []
            tgt_labels = []
            tgt_no_labels = []
            tgt_no_features = []

            train_tgt_data_in, train_tgt_data_notin = self.get_weak_label_v2(train_tgt_data)
            if self.hparams.is_get_clean_data:
                new_index = train_tgt_data_notin['index_new'].values.tolist()
                zero_index = [i for i in new_index if train_tgt_label[i] == 0][:int(self.hparams.clean_count / 2)]
                one_index = [i for i in new_index if train_tgt_label[i] == 1][:int(self.hparams.clean_count / 2)]
                index = zero_index + one_index
                train_tgt_data = pd.DataFrame(train_tgt_data, columns=['encoded_text', 'index', 'domain'])
                select_data = train_tgt_data.iloc[index, :]

                select_tgt_features = select_data[['encoded_text', 'domain']].values.tolist()
                select_tgt_labels = [train_tgt_label[i] for i in index]
                assert len(select_tgt_labels) == self.hparams.clean_count
                assert len(select_tgt_features) == self.hparams.clean_count
                # src_features += select_tgt_features
                # src_labels += select_tgt_labels
                if self.hparams.model_type != "new":
                    src_features += select_tgt_features
                    src_labels += select_tgt_labels
                else:
                    tgt_features += select_tgt_features
                    tgt_labels += select_tgt_labels
            if self.hparams.model_type == "new":
                tgt_features += train_tgt_data_in[['encoded_text', 'domain']].values.tolist()
                tgt_labels += train_tgt_data_in['weak_label'].tolist()
                if debug:
                    tgt_no_labels = list(tgt_labels)
                    tgt_no_features = list(tgt_features)
                    print("DEBUGGING")
                else:
                    tgt_no_labels += train_tgt_label
                    tgt_no_features += [[i[0], i[2]] for i in train_tgt_data]

                print("With Weak Samples")
            else:
                # ['encoded_text', 'index', 'domain']
                tgt_features += [[i[0], i[2]] for i in train_tgt_data]
                tgt_labels += train_tgt_label

            if self.hparams.is_flip_label:
                src_features += tgt_features
                src_labels += tgt_labels




        elif type == 'test':
        # else:
           # Only tgt dataset for test
            src_features = [(i[0], i[2]) for i in test_tgt_data] #test_tgt_data
            src_labels = test_tgt_label
            tgt_features = [(i[0], i[2]) for i in test_tgt_data]
            tgt_labels = test_tgt_label
        else:
            # Only tgt dataset for validation
            src_features = [(i[0], i[2]) for i in val_tgt_data]
            src_labels = val_tgt_label
            tgt_features = [(i[0], i[2]) for i in val_tgt_data]
            tgt_labels = val_tgt_label



        if type == "train":
            if hparams.model_type == 'new':
                max_len = max(len(src_features), len(tgt_features), len(tgt_no_features))
                if len(src_features) < max_len:
                    src_features = src_features * int(max_len / len(src_features))
                    src_labels = src_labels * int(max_len / len(src_labels))
                    src_features = src_features + src_features[:max_len - len(src_features)]
                    src_labels = src_labels + src_labels[:max_len - len(src_labels)]
                if len(tgt_features) < max_len:
                    tgt_features = tgt_features * int(max_len / len(tgt_features))
                    tgt_labels = tgt_labels * int(max_len / len(tgt_labels))
                    tgt_features = tgt_features + tgt_features[:max_len - len(tgt_features)]
                    tgt_labels = tgt_labels + tgt_labels[:max_len - len(tgt_labels)]
                if len(tgt_no_features) < max_len:
                    tgt_no_features = tgt_no_features * int(max_len / len(tgt_no_features))
                    tgt_no_labels = tgt_no_labels * int(max_len / len(tgt_no_labels))
                    tgt_no_features = tgt_no_features + tgt_no_features[:max_len - len(tgt_no_features)]
                    tgt_no_labels = tgt_no_labels + tgt_no_labels[:max_len - len(tgt_no_labels)]


            else:
                max_len = max(len(src_features), len(tgt_features))
                if len(src_features) < max_len:
                    src_features = src_features * int(max_len / len(src_features))
                    src_labels = src_labels * int(max_len / len(src_labels))
                    src_features = src_features + src_features[:max_len - len(src_features)]
                    src_labels = src_labels + src_labels[:max_len - len(src_labels)]
                else:
                    tgt_features = tgt_features * int(max_len / len(tgt_features))
                    tgt_labels = tgt_labels * int(max_len / len(tgt_labels))
                    tgt_features = tgt_features + tgt_features[:max_len - len(tgt_features)]
                    tgt_labels = tgt_labels + tgt_labels[:max_len - len(tgt_labels)]


        self.src_features = [i[0] for i in src_features]
        self.src_domain = [i[1] for i in src_features]
        self.src_labels = src_labels
        self.tgt_features = [i[0] for i in tgt_features]
        self.tgt_domain = [i[1] for i in tgt_features]
        self.tgt_labels = tgt_labels
        self.flag_no = False
        if self.hparams.model_type == "new" and type == "train":
            self.tgt_no_labels = tgt_no_labels
            self.tgt_no_features = [i[0] for i in tgt_no_features]
            self.tgt_no_domain = [i[1] for i in tgt_no_features]
            self.flag_no = True


    def get_number_data(self, features, labels, number):
        return features[:number], labels[:labels]

    def get_domain_data(self, self_data, domain, random_seed, is_not_in=False):
        data = []
        print("There are {} raw data".format(len(self_data)))
        if is_not_in is False and type(domain) is not list and "," not in domain:
            for i in self_data:
                if domain in i['domain']:
                    data.append(i)

            # balance the dataset
            print("there are {} samples for {}".format(len(data), domain))
            one = [(i['encode_text'], i['index']) for i in data if i['label'] == 1]
            zero = [(i['encode_text'], i['index']) for i in data if i['label'] == 0]
            print("there are {} zeros and {} ones".format(len(one), len(zero)))
            min_len = min(len(one), len(zero))
            one = one[:min_len]
            zero = zero[:min_len]
            data = one + zero
            data = [(*i, domain_map.get(domain, 2)) for i in data]
            # train_X, train_Y, test_X, test_Y, val_X, val_Y
            print("{}: {}".format(domain, len(data)))
            train_data, train_label, test_data, test_label, val_data, val_label = data_split_val(data, [1] * min_len + [
                0] * min_len)
            print("Domain: {} Train Size: {} Test Size {}".format(domain, len(train_data), len(test_data)))
            return train_data, train_label, test_data, test_label, val_data, val_label
        else:

            train_data_list = []
            train_label_list = []
            test_data_list = []
            test_label_list = []
            val_data_list = []
            val_label_list = []
            # domains = list(set([i['domain'] for i in self_data]))
            if is_not_in:
                domains = list(domain_map.keys())
                domains = [i for i in domains if domain not in i]
            else:
                if type(domain) is str and "," in domain:
                    domains = domain.split(",")
                else:
                    domains = domain

            for domain in domains:
                train_data, train_label, test_data, test_label, val_data, val_label = self.get_domain_data(self_data,
                                                                                                           domain,
                                                                                                           random_seed=random_seed)

                train_data_list += train_data
                train_label_list += train_label
                test_data_list += test_data
                test_label_list += test_label
                val_data_list += val_data
                val_label_list += val_label
            return train_data_list, train_label_list, test_data_list, test_label_list, val_data_list, val_label_list

    def label_probability(self, train_tgt_label):
        _, counts = torch.unique(train_tgt_label, sorted=True, return_counts=True)
        label_pro = counts.float() / torch.sum(counts)
        return label_pro

    def get_weak_labels(self, tgt_train_data):


        data = pd.read_csv(self.hparams.weak_labels_path, header=None)
        data = data.rename(columns={0:"index",1:'weak_label'})
        tgt_train_data = pd.DataFrame(tgt_train_data, columns=['encoded_text','index', 'domain'])
        tgt_train_data['index_new'] = list(range(len(tgt_train_data)))
        tgt_train_data_in = tgt_train_data.join(data.set_index('index'), how='inner', on='index')
        tgt_train_data_notin = tgt_train_data[tgt_train_data['index'].apply(lambda x: x not in set(data['index'].values.tolist()))]
        print("There are {} weak samples".format(len(tgt_train_data_in)))
        tgt_train_data_in['weak_label'] = tgt_train_data_in['weak_label'].apply(lambda x: int(x))
        tgt_train_data_in = tgt_train_data_in[['encoded_text','weak_label',"domain"]]

        return tgt_train_data_in, tgt_train_data_notin


    def get_weak_label_v2(self, tgt_train_data):
        data = pd.read_csv(self.hparams.weak_labels_path)
        tgt_train_data = pd.DataFrame(tgt_train_data, columns=['encoded_text', 'index', 'domain'])
        data = data.iloc[tgt_train_data['index'], :]
        weak_labels = data[self.hparams.weak_fn].tolist()

        zero_index = set(data.iloc[list(np.argsort(weak_labels)[:self.hparams.weak_label_count]),:]['index'].tolist())
        one_index = set(data.iloc[list(np.argsort(weak_labels)[-self.hparams.weak_label_count:])]['index'].tolist())
        tgt_train_data['index_new'] = list(range(len(tgt_train_data)))
        def helper_fn(x):
            if x in zero_index:
                return 0
            elif x in one_index:
                return 1
            else:
                return np.nan
        tgt_train_data['weak_label'] = tgt_train_data['index'].apply(lambda x: helper_fn(x))
        tgt_train_data_notin = tgt_train_data[tgt_train_data['weak_label'].isna()]
        tgt_train_data_in = tgt_train_data[tgt_train_data['weak_label'].notna()]
        tgt_train_data_in = tgt_train_data_in[['encoded_text', 'weak_label', "domain"]]
        tgt_train_data_in['weak_label'] = tgt_train_data_in['weak_label'].astype('int')

        return tgt_train_data_in, tgt_train_data_notin
    def __len__(self):
        return len(self.src_features)

    def __getitem__(self, item):
        # return torch.tensor(self.labels[item], dtype=torch.long), torch.tensor(self.features[item], dtype=torch.long)

        output = (torch.tensor(self.src_labels[item]), torch.tensor(self.src_domain[item]), torch.tensor(self.src_features[item][0]),
        torch.tensor(self.tgt_labels[item]), torch.tensor(self.tgt_domain[item]), torch.tensor(self.tgt_features[item][0]), )

        if self.flag_no:
            output += (torch.tensor(self.tgt_no_labels[item]), torch.tensor(self.tgt_no_domain[item]),
                       torch.tensor(self.tgt_no_features[item][0]),)


        return output

class EANNTextDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, type, tokenizer):
        super(EANNTextDataset, self).__init__()
        self.hparams = hparams
        if hparams.is_not_in:
            file_name = "./data/EANN_s:gossip,politi_t:health_deterrent.torch"
        else:
            if hparams.tgt_domain == "gossip" and hparams.src_domain == "health_deterrent,politi":
                file_name = f"data/EANN_s:gossip,politi_t:health_deterrent.torch"
            elif hparams.tgt_domain == "politi" and hparams.src_domain == "gossip,health_deterrent":
                file_name = f"data/EANN_s:gossip,health_deterrent_t:politi.torch"
            else:
                file_name = f"data/EANN_s:{hparams.src_domain}_t:{hparams.tgt_domain}.torch"
        if os.path.exists(file_name):
            data = torch.load(file_name)
            for key, value in data.items():
                setattr(self, key, json.loads(value))
        else:
            if tokenizer is None:
                tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            csv_filename = file_name.replace("torch",'csv')
            data = pd.read_csv(csv_filename)

            data['encode_text'] = data['content'].apply(lambda x: tokenizer.encode(x,
                                                                                   max_length=hparams.max_length,
                                                                                   truncation=True,
                                                                                   pad_to_max_length=True, ))

            self.data = data.to_json(orient="records")
            torch.save({"data": self.data
                        }, file_name)
        self.random_seed = hparams.random_seed
        src_domain = hparams.src_domain
        if "," in src_domain:
            src_domain = src_domain.split(",")
        tgt_domain = hparams.tgt_domain
        is_not_in = hparams.is_not_in
        train_src_data, train_src_label, test_src_data, test_src_label, val_src_data, val_src_label = self.get_domain_data(self.data, src_domain,
                                                                                                self.random_seed, is_not_in)

        train_tgt_data, train_tgt_label, test_tgt_data, test_tgt_label, val_tgt_data, val_tgt_label = self.get_domain_data(self.data,
                                                                                                    tgt_domain, random_seed=self.random_seed)


        if src_domain != tgt_domain and type == 'train' and hparams.is_few_shot:
            train_data_tgt, train_label_tgt, _, _ = self.get_domain_data(self.data, tgt_domain, random_seed=self.random_seed)
            hparams.tgt_train_size = float(hparams.tgt_train_size)
            max_len = hparams.tgt_train_size if hparams.tgt_train_size > 1 else int(
                hparams.tgt_train_size * len(train_data_tgt))
            max_len = int(max_len)
            train_src_data += train_data_tgt[:max_len]
            train_src_label += train_label_tgt[:max_len]
        elif src_domain == tgt_domain and type == 'train':
            max_len = hparams.tgt_train_size if hparams.tgt_train_size > 1 else int(
                hparams.tgt_train_size * len(train_src_data))
            max_len = int(max_len)
            train_src_data = train_src_data[:max_len]
            train_src_label = train_src_label[:max_len]

        if type == "train":
            src_features = train_src_data
            src_labels = train_src_label
            tgt_features = []
            tgt_labels = []

            tgt_features += [[i[0], i[1]] for i in train_tgt_data]
            tgt_labels += train_tgt_label

        elif type == 'test':
           # Only tgt dataset for test
            src_features = test_tgt_data
            src_labels = test_tgt_label
            tgt_features = test_tgt_data
            tgt_labels = test_tgt_label
        else:
            # Only tgt dataset for validation
            src_features = val_tgt_data
            src_labels = val_tgt_label
            tgt_features = val_tgt_data
            tgt_labels = val_tgt_label

        if type == "train":
            max_len = max(len(src_features), len(tgt_features))
            if len(src_features) < max_len:
                src_features = src_features * int(max_len / len(src_features))
                src_labels = src_labels * int(max_len / len(src_labels))
                src_features = src_features + src_features[:max_len - len(src_features)]
                src_labels = src_labels + src_labels[:max_len - len(src_labels)]
            else:
                tgt_features = tgt_features * int(max_len / len(tgt_features))
                tgt_labels = tgt_labels * int(max_len / len(tgt_labels))
                tgt_features = tgt_features + tgt_features[:max_len - len(tgt_features)]
                tgt_labels = tgt_labels + tgt_labels[:max_len - len(tgt_labels)]



        self.src_features = [i[0] for i in src_features]
        self.src_domain = [i[1] for i in src_features]
        self.src_labels = src_labels
        self.tgt_features = [i[0] for i in tgt_features]
        self.tgt_domain = [i[1] for i in tgt_features]
        self.tgt_labels = tgt_labels

    def get_weak_labels(self, tgt_train_data):
        data = pd.read_csv(self.hparams.weak_labels_path, header=None)
        data = data.rename(columns={0: "index", 1: 'weak_label'})
        tgt_train_data = pd.DataFrame(tgt_train_data, columns=['encoded_text', 'domain', 'index'])
        tgt_train_data['index_new'] = list(range(len(tgt_train_data)))
        tgt_train_data_in = tgt_train_data.join(data.set_index('index'), how='inner', on='index')
        tgt_train_data_notin = tgt_train_data[
            tgt_train_data['index'].apply(lambda x: x not in set(data['index'].values.tolist()))]
        print("There are {}".format(len(tgt_train_data_in)))
        tgt_train_data_in['weak_label'] = tgt_train_data_in['weak_label'].apply(lambda x: int(x))
        tgt_train_data_in = tgt_train_data_in[['encoded_text', 'weak_label', "domain"]]

        return tgt_train_data_in, tgt_train_data_notin

    def get_domain_data(self, self_data, domain, random_seed, is_not_in=False):
        data = []
        if is_not_in is False and type(domain) is not list:
            for i in self_data:
                if domain in i['domain']:
                    data.append(i)

            # balance the dataset
            one = [(i['encode_text'], i['cluster_id'], i['Unnamed: 0']) for i in data if i['label'] == 1]
            zero = [(i['encode_text'], i['cluster_id'], i['Unnamed: 0']) for i in data if i['label'] == 0]
            min_len = min(len(one), len(zero))
            one = one[:min_len]
            zero = zero[:min_len]
            data = one + zero
            # data = [(i, domain_map.get(domain, 2)) for i in data]
            # train_X, train_Y, test_X, test_Y, val_X, val_Y
            train_data, train_label, test_data, test_label, val_data, val_label = data_split_val(data, [1] * min_len + [
                0] * min_len, random_seed=random_seed)
            return train_data, train_label, test_data, test_label, val_data, val_label
        else:
            train_data_list = []
            train_label_list = []
            test_data_list = []
            test_label_list = []
            val_data_list = []
            val_label_list = []
            # domains = list(set([i['domain'] for i in self_data]))
            if is_not_in:
                domains = list(domain_map.keys())
                domains = [i for i in domains if domain not in i]
            else:
                domains = domain

            for domain in domains:
                train_data, train_label, test_data, test_label,  val_data, val_label = self.get_domain_data(self_data, domain, random_seed=random_seed)
                print("Domain: {} Train Size: {} Test Size {}".format(domain, len(train_data), len(test_data)))
                train_data_list += train_data
                train_label_list += train_label
                test_data_list += test_data
                test_label_list += test_label
                val_data_list += val_data
                val_label_list += val_label
            return train_data_list, train_label_list, test_data_list, test_label_list, val_data_list, val_label_list

    def __len__(self):
        return len(self.src_features)

    def __getitem__(self, item):
        # return torch.tensor(self.labels[item], dtype=torch.long), torch.tensor(self.features[item], dtype=torch.long)
        th = (torch.LongTensor([self.src_labels[item]]), torch.LongTensor(self.src_domain[item]), torch.tensor(self.src_features[item]),
        torch.LongTensor([self.tgt_labels[item]]), torch.LongTensor(self.tgt_domain[item]), torch.tensor(self.tgt_features[item]))
        return torch.tensor(int(self.src_labels[item])), torch.tensor(int(self.src_domain[item])), torch.tensor(self.src_features[item]), \
        torch.tensor(int(self.tgt_labels[item])), torch.tensor(int(self.tgt_domain[item])), torch.tensor(self.tgt_features[item])

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, type, tokenizer):
        data = []
        file_name = "./data/text.torch"
        if os.path.exists(file_name) and False:
            data = torch.load(file_name)
            for key, value in data.items():
                setattr(self, key, value)
        else:
            with open(hparams.data_path,'r') as f1:
                for line in f1.readlines():
                    line = json.loads(line)
                    label = line['label']
                    label = 1 if label == 'fake' else 0
                    lang = line['lang']
                    text = line['text']
                    if len(text.split()) > 50:
                        continue
                    if "# GiselleMaxwellslist became unsealed some time" in text:
                        continue
                    text = data_preprocess(text)
                    text = " ".join(text)
                    news_id = line['news_id']


                    data.append((label, lang, text, news_id))
                    self.data = data
            torch.save({"data":self.data}, file_name)
        self.random_seed = hparams.random_seed

        self.data = [i for i in self.data if i[1] == 'en']

        one = [i for i in self.data if i[0] == 1]
        zero = [i for i in self.data if i[0] == 0]
        min_len = min(len(one), len(zero))
        one = one[:min_len]
        zero = zero[:min_len]
        self.data = one+zero
        train_data, _, test_data, _ = data_split(self.data, [i[0] for i in self.data], random_seed=self.random_seed)

        if type == "train":

            self.features = [i[2] for i in train_data]
            self.labels = [i[0] for i in train_data]

        else:
            self.features = [i[2] for i in test_data]
            self.labels = [i[0] for i in test_data]



        # tokenize the text
        self.features = [tokenizer.encode(i,
                                  max_length=hparams.max_length,
                                  truncation=True,
                                  pad_to_max_length=True,
                                  ) for i in self.features]
        assert all([len(i) == hparams.max_length for i in self.features])
        # print(len(self.features[0]))
        # self.features = [i[0] for i in self.features]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return torch.tensor(self.labels[item], dtype=torch.long), torch.tensor(self.features[item], dtype=torch.long)

class CommenetDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, type, tokenizer):
        self.hparams = hparams
        store_file_name = "./data_comment.torch"
        if os.path.exists("./data_comment.torch") and self.hparams.overwrite is False:
            data = torch.load(store_file_name)
            for key, value in data:
                setattr(self, key, value)
        else:
            news_content_list = []
            comments_list = []
            users_list = []
            labels_list = []
            with open(hparams.data_path,'r') as f1:
                for line in f1.readlines():
                    line = json.loads(line)
                    label = line['label']
                    label = 1 if label == 'fake' else 0
                    comments = []
                    times = []
                    users = []
                    for i in line['social']:
                        try:
                            created_at = i['created_at']
                            created_at = datetime.fromtimestamp(i / 1000) if type(created_at) is Int64 else dateparser.parse(created_at)
                            times.append((len(times), created_at))

                            if "tweet" in i.keys():
                                comment = i['tweet']
                            elif "text" in i.keys():
                                comment = i['text']
                            elif "raw" in i.keys():
                                comment = i['raw']['tweet']
                            elif "full_text" in i.keys():
                                comment = i['full_text']
                            else:
                                continue

                            if "username" in i.keys():
                                users.append(i['username'])
                            elif "screen_name" in i.keys():
                                users.append(i['screen_name'])
                            elif "user" in i.keys():
                                users.append(i['user']['screen_name'])
                            else:
                                continue

                            times.append(created_at)
                            comments.append(comment)
                        except:
                            continue

                    if len(times) > 0:
                        order = sorted(times, key=lambda x:x[1])
                        order = [order[0] for i in order]
                        comments = [comments[i] for i in order]
                        users = [users[i] for i in order]

                    comments_list.append(comments)
                    users_list.append(users)
                    news_content_list.append(line['text'])
                    labels_list.append(label)

            # split the news content into different sentences
            news_sentences = [[j.text for j in nlp(i).sents] for i in news_content_list]
            news_sentences = [[j for j in i if len(j) > 10] for i in news_sentences]
            # truncate the senteces
            sentence_count = hparams.sentence_count
            comment_count = hparams.comment_count
            vectorizer = CountVectorizer(lowercase=False)

            users_list_std = [" ".join(i) for i in users_list if len(i) > 0]
            users_matrix = vectorizer.fit_transform(users_list_std)
            users_matrix = users_matrix.transpose()
            u, _, _ = svds(users_matrix, k=10)
            self.u = np.vstack((u, [0.] * u.shape[1]))


            news_sentences = [i[:sentence_count] + (sentence_count - len(i))
                              * ["<pad>"] for i in news_sentences]
            comments_list = [[j for j in i if len(j) > 3] for i in comments_list]
            comments_list = [i[:comment_count] + (comment_count - len(i)) * ["<pad>"]
                        for i in comments_list]
            users_list = [[vectorizer.vocabulary_[j] for j in i[:comment_count]] + (comment_count - len(i)) * [u.shape[1] - 1] for i in users_list]




            features = list(zip(news_sentences, comments_list, users_list))
            self.features_all = features
            self.labels_all = labels_list
            save_dic = {
                        "features": self.features,
                        "labels": self.labels,
                        "u": self.u}
            torch.save(save_dic, store_file_name)

        self.random_seed = hparams.random_seed
        train_X, train_Y, test_X, test_Y = \
            data_split(self.features_all, self.labels_all, random_seed=self.random_seed)

        if type == "train":
            print("Stop")
            self.features = train_X
            self.labels = train_Y
        else:
            print("Hello")
            self.features = test_X
            self.labels = test_Y

        # tokenize the text
        self.news_content = [
            i[0]
            for i in self.features
        ]

        self.news_content = [
            # self.debug_func(i, tokenizer)
            [tokenizer.encode(j.strip(), max_length=hparams.max_sentence_length,
                               truncation=True,
                               pad_to_max_length=True)
                for j in i]
            for i in self.news_content
        ]

        self.comments = [
            i[1]
            for i in self.features
        ]
        self.comments = [
            [tokenizer.encode(j, max_length=hparams.max_comment_length,
                               truncation=True,
                               pad_to_max_length=True,)
             for j in i
            ]
            for i in self.comments
        ]

        self.users = [
            i[2]
            for i in self.features
        ]





        # print(len(self.features[0]))
        # self.features = [i[0] for i in self.features]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return_tuple = (torch.tensor(self.labels[item], dtype=torch.long),
               torch.tensor(self.news_content[item], dtype=torch.long),
               torch.tensor(self.comments[item], dtype=torch.long))
        if self.hparams.use_social:
            return_tuple += (torch.tensor(self.u[self.users[item]], dtype=torch.float),)




    def debug_func(self, i, tokenizer):
        for index, j in enumerate(i):
            tokenizer.encode(j.strip(), max_length=self.hparams.max_sentence_length,
                             truncation=True,
                             pad_to_max_length=True)

            try:
                tokenizer.encode(j.strip(), max_length=self.hparams.max_sentence_length,
                              truncation=True,
                              pad_to_max_length=True)
            except:
                print(index)
                print(j)
                print(i)
                exit()


from collections import namedtuple

if __name__ == '__main__':
    args = namedtuple('args',['data_path'])
    data = SimpleTextDataset(None,"train",'politic', None)