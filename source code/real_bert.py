import bert
import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf
import numpy as np
#########################
#implement Bert model on LIAR dataset
#achieve acc > 25%
#other metrics: prec, recall, f1, support
# [1] so the blog implementation uses binary lablel...How to handle multi - label??
# we need to make this work 
##########################

#print(tf.__version__)

#upload the dataset
train_bert_data = pd.read_csv('C:/Users/farid/PycharmProjects/CS7992/DB/train01.csv')
test_bert_data = pd.read_csv("C:/Users/farid/PycharmProjects/CS7992/DB/test01.csv")

print("train size: " , train_bert_data.shape)
print("test size: " , test_bert_data.shape)

#[LABEL TEXT]
#print(train_bert_data.columns.values)

#LABEL: FALSE, TRUE, half-true, mostly-true, barely-true, pants-fire
#print(train_bert_data.LABEL.unique())

##how to handle multi- label?
y = train_bert_data['TEXT']
#y = np.array(list(map(lambda x: 0 if x=="FALSE" else 0,y)))



#####BERT tokenization#############
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4", trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
#############################################
#lumpar = tokenizer.tokenize("Bring back Trump Again.com senora")
#print(lumpar)
#lumpar01=tokenizer.convert_tokens_to_ids(tokenizer.tokenize("dont be so judgmental"))
#print(lumpar01)
#####################################################


