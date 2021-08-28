##################
#This is to implement Bert on lIar dataset
#################
#################

import pandas as pd
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

#load dataset
train_set = pd.read_csv('C:\\Users\\fksch\\Desktop\\CS7999\\archive\\liar_dataset\\train.csv')
test_Set = pd.read_csv('C:\\Users\\fksch\\Desktop\\CS7999\\archive\\liar_dataset\\test.csv')

#print (train_set['LABEL'].value_counts())

#turn the labels into numeric form
possible_labels = train_set.LABEL.unique()

label_dict = {}
for index, possible_labels in enumerate(possible_labels):
    label_dict[possible_labels] = index
label_dict

#train-test split
from sklearn.model_selection import train_test_split





