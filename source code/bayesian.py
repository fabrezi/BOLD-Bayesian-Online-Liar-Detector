##############################################
#this code is to perfrom supervised learn with sklearn package on the 
#dataset. 

########################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
###################################################
tf = pd.read_csv('C:\\Users\\farid\\PycharmProjects\\CS7992\\DB\\train01.csv')
testdata = pd.read_csv('C:\\Users\\farid\\PycharmProjects\\CS7992\\DB\\test01.csv')

#form each label to numeric value
col = ['LABEL', 'TEXT']
tf = tf[col]
tf = tf[pd.notnull(tf['TEXT'])]
tf.columns = ['LABEL', 'TEXT']
tf['category_id'] = tf['LABEL'].factorize()[0]
category_id_tf = tf[['LABEL', 'category_id']].drop_duplicates().sort_values('category_id')

#explain
category_to_id = dict(category_id_tf.values)
id_to_category = dict(category_id_tf[['category_id', 'LABEL']].values)
print(tf.head())

print("train size:" ,tf.shape)
print("test size:" , testdata.shape)

def check_dist(dataset):
    sns.countplot(x='LABEL', data=tf, palette='hls')
asd = pd.DataFrame(tf['LABEL'])
check_dist(tf)
plt.show()

check_dist(testdata)
plt.show() #work

#convert text to vector
#Tf-idf part
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1'
                        , ngram_range=(1,2), stop_words='english')
features= tfidf.fit_transform(tf.TEXT).toarray()
labels = tf.category_id

#fit the train set
X_train, X_test, y_train, y_test = train_test_split(tf['TEXT'], tf['LABEL'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

print("ok")
#test- bracket
print(clf.predict(count_vect.transform(["The Fed created $1.2 trillion out of nothing, gave it to banks, and some of them foreign banks, so that they could stabilize their operations."])))

#model formulation
models = [RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
          LinearSVC(),
          MultinomialNB(),
          LogisticRegression(random_state=0, max_iter=1000)]
CV = 5
cv_tf = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_tf = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_tf)
sns.stripplot(x='model_name', y='accuracy', data=cv_tf,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
print("ok")

vlad = cv_tf.groupby('model_name').accuracy.mean()
print("classifier    accuracy_Avg:", vlad)

model = MultinomialNB()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features,
                                                                                 labels, tf.index,
                                                                                 test_size=0.33,
                                                                                 random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=category_id_tf.LABEL.values, yticklabels=category_id_tf.LABEL.values)
plt.ylabel('Actual')
plt.xlabel('predicted')
plt.show()

#print(y_test)
#print(y_pred)
#print(tf['LABEL'].unique())
#performance report
print(metrics.classification_report(y_test, y_pred, target_names=tf['LABEL'].unique()))






