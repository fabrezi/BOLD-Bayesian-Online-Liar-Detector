import bert
import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#########################
#implement Bert model on LIAR dataset
#achieve acc > 25%
#other metrics: prec, recall, f1, support
# [1] so the blog implementation uses binary lablel...How to handle multi - label??
#
##########################

#print(tf.__version__)

#upload the dataset
train_bert_data = pd.read_csv('C:/Users/farid/PycharmProjects/CS7992/DB/train01.csv')
test_bert_data = pd.read_csv("C:/Users/farid/PycharmProjects/CS7992/DB/test01.csv")

print("train size: ", train_bert_data.shape)
print("test size: ", test_bert_data.shape)

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
#########################################################################################

#create the model
class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output
###############################################################################################

#hyperparameters
VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2

DROPOUT_RATE = 0.2

NB_EPOCHS = 5

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)

if OUTPUT_CLASSES == 2:
    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])

text_model.fit(train_bert_data, epochs=NB_EPOCHS)

results = text_model.evaluate(test_bert_data)
print(results)






