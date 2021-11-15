from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_score, recall_score
import numpy as np
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import torch
import numpy as np
import random
from ray import tune
import preprocessor as p

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
try:
    import twint
except:
    print("If not install twint, you cannot utilize defend for inference. ")

import string
import yaml
## Read Configuration file

def hyper_parameter_search(config_dict):
    def locate_grid_search(input_dict):
        parameter_config = None
        for key, value in input_dict.items():
            if key == "grid_search":
                parameter_config = {
                    hyper: tune.grid_search(value)
                    for hyper, value in value.items()
                }
                return parameter_config
            elif key == "bayesian":
                parameter_config = {

                }
            elif type(value) is dict:
                parameter_config = locate_grid_search(value)
        return parameter_config

    config = locate_grid_search(config_dict)
    search_hyper = list(config.keys())

    return config, search_hyper

def flat_dict(config_dict):
    def remove_grid_search(input_dict):
        parameter_config = {}
        for key, value in input_dict.items():
            if key == "grid_search":
                continue
            elif type(value) is dict:
                parameter_config = {**remove_grid_search(value), **parameter_config}
            else:
                parameter_config[key] = value
        return parameter_config

    config = remove_grid_search(config_dict)
    return config
def extract_yaml(config_path):
    def check_yaml_file(doc):
        new_doc = {}
        for key, value in doc.items():
            if type(value) is dict:
                new_value = check_yaml_file(value)
            elif type(value) is str and ".yml" in value:
                new_value = extract_yaml(value)
            else:
                new_value = value
            new_doc[key] = new_value
        return new_doc

    with open(config_path) as f1:
        docs = yaml.load_all(f1, Loader=yaml.FullLoader)
        doc = next(docs)
        doc = check_yaml_file(doc)


    return doc
def read_yaml(config_path):
    doc = extract_yaml(config_path)
    tune_config, search_parameters = hyper_parameter_search(doc)
    general_config = flat_dict(doc)
    return tune_config, search_parameters, general_config

def data_split(features, labels, test_ratio=0.2, random_seed=123):
    train_X, test_X, train_Y, test_Y = train_test_split(features, labels,
                                                        test_size=test_ratio, random_state=random_seed, stratify=labels)
    return train_X, train_Y, test_X, test_Y

def data_split_val(features, labels, test_ratio=0.2, val_ratio=0.1, random_seed=123):
    print("Random Seed is {}".format(random_seed))
    train_X, test_X, train_Y, test_Y = train_test_split(features, labels,
                                                        test_size=test_ratio, random_state=random_seed, stratify=labels)
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=val_ratio/(1-test_ratio), random_state=random_seed, stratify=train_Y)
    # print("# ATTENTION: We only keep small amount of validation dataset 5%")
    val_X = val_X[:int(len(val_X)/2)]
    val_Y = val_Y[:int(len(val_Y)/2)]
    # ele = int(len(val_X) / 5)
    # val_one = list(np.array(val_X)[np.array(val_Y) == 0][:ele])
    # val_zero = list(np.array(val_X)[np.array(val_Y) == 1][:ele])
    # val_X = val_one + val_zero
    # val_Y = [0] * ele + [1] * ele

    return train_X, train_Y, test_X, test_Y, val_X, val_Y

def evaluation(logits, test_Y):
    predict_Y = np.argmax(logits, axis=1)
    acc = accuracy_score(y_pred=predict_Y, y_true=test_Y)

    try:
        # positive f1:
        average = "binary"
        pos_f1, pos_precision, pos_recall = seperate_score(predict_Y, test_Y, average)
        # negative f1:
        predict_Y = 1 - predict_Y
        test_Y = 1 - test_Y
        neg_f1, neg_precision, neg_recall = seperate_score(predict_Y, test_Y, average)
        return {"acc": acc,
                "positive_f1": pos_f1, "pos_recall": pos_recall, "pos_precision": pos_precision,
                "neg_f1": neg_f1, "neg_recall": neg_recall, "neg_precision": neg_precision}

    except:
        average = "micro"
        f1, precision, recall = seperate_score(predict_Y, test_Y, average)
        return {"acc": acc, "f1": f1, "recall": recall, "precision": precision}

    # return {"acc":acc, "f1":f1, "cf_m":cf_m}



def seperate_score(predict_Y, test_Y, average):
    f1 = f1_score(y_pred=predict_Y, y_true=test_Y,average=average)
    cf_m = confusion_matrix(y_pred=predict_Y, y_true=test_Y)
    precision = precision_score(y_true=test_Y, y_pred=predict_Y, average=average)
    recall = recall_score(y_true=test_Y, y_pred=predict_Y, average=average)
    return f1, precision, recall


def set_random_seed(seed):
    torch.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

def get_tweets(search_query):
    # remove punctuation
    new_query = str(search_query).translate(str.maketrans('', '', string.punctuation))
    c = twint.Config()
    c.Search = new_query
    c.Store_pandas = True
    c.Pandas_clean = True
    twint.run.Search(c)
    data = twint.storage.panda.Tweets_df
    comments = data.loc[:, 'text'].values.tolist()
    return comments

def data_preprocess(text):
    # tokenization and remove punctuation
    #
    text = p.clean(text)
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        print(str(e))
        # print(text)

    words = [word.lower() for word in tokens if word.isalnum()]

    # remove stopwords
    #
    stop_words = set(stopwords.words('english'))
    stop_words.update(['claims','post','video','no','not',
                       'photo','show','say','facebook','that','image','claim','people','picture'])
    # stopwords.update(['says','The','virus','day',"COVID-19", 'COVID-19.', 'covid-19', "COVID-19,", 'covid-19', 'the coronavirus', 'COVID', 'the','show', 'said','new', 'people', 'say', 'coronavirus', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    # words = [i for i in words if not i.lower() in stop_words]

    # lemmatize
    #
    lemmatizer = WordNetLemmatizer()
    try:
        words = [lemmatizer.lemmatize(w, pos="v") for w in words]
    except:
        print(words)

    # remove stopwords once more after lemmatization
    words = [i for i in words if not i in stop_words]

    return words


if __name__ == '__main__':
    th = read_yaml("config/MDWS.yml")
    # th = hyper_parameter_search(th)
    print(th)