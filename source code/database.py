##we need to get pretty figs for dataset section
##bar graph for train and test data

import pandas as pd
import matplotlib.pyplot as plt

#upload dataset
train_data_fig = pd.read_csv('C:\\Users\\farid\\PycharmProjects\\CS7992\\DB\\train01.csv')
test_data_fig = pd.read_csv('C:\\Users\\farid\\PycharmProjects\\CS7992\\DB\\test01.csv')

plt.hist(train_data_fig['LABEL'], bins=15)
plt.xlabel('tags')
plt.ylabel('frequency')
plt.show()

plt.hist(test_data_fig['LABEL'], bins=15)
plt.xlabel('tags')
plt.ylabel('frequency')
plt.show()




#plt.figure(figsize=(20,5))
plt.hist(train_data_fig['PARTY-AFFILIATION'].astype(str), bins=100, align='mid')
plt.subplots_adjust(bottom=0.5)
#plt.rcParams.update({'font.size': 12})
plt.xticks(rotation=90)
plt.xlabel('speaker-state-info')
plt.show()

(trainer['SPEAKER'].value_counts().sort_values(ascending=False).head(50).plot.bar())
plt.subplots_adjust(bottom=0.5)
plt.xlabel('speaker')
plt.ylabel('frequency')
plt.show()

