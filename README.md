# Multi-Domain Fake News Classification

### Dataset Location
You can find the dataset at the appendix. 


### Install the requirement
```shell script
pip install -r requirements.txt
# install spacy English Module
python -m spacy download en_core_web_sm
# store the tensorboard result and model checkpoints
mkdir "tb_logs"
```

### Usage
```shell script
python --tgt_domain gossip \
--src_domain politi,health_deterrent \
--epochs 50 \
--weak_labels_path ./data/weak_label_all.csv \
--pre_train_epochs 20 \
--special_tag "weight_analysis" \
--model_type new \
--is_omit_logits \
--weak_fn adverb \
--weight_decay 0 \
--main_lr_rate 0.001 \
--group_lr_rate 0.01 \
--lr_rate 0.001 \
--hyper_beta 0.3 \
--lambda 0.1 \
--weak_label_count 50
```
You can change the target domain and source domain based on your need. There is no order requirement for source domains. 

There are three different weak labeling functions: __you__, __adverb__ and __swear__.
 





