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

