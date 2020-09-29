from sklearn.decomposition import PCA

print(__doc__)

def sound():
    from beepy import beep
    beep(sound=1)  # integer as argument

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import main as m
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import ngrams
# #############################################################################
# Generate sample data
import os.path
from os import path
if path.exists("../data/dataframe.csv"):
    originalDF = pd.read_csv('../data/dataframe.csv', encoding='utf-8')
else:
    data = m.getData()
    id = []
    age = []
    gender = []
    grams = []
    posTagVectorizer = []
    for entry in data.keys():
        if data[entry].age != "":
            id.append(int(entry))
            age.append(int(data[entry].age))
            if data[entry].gender == "male":
                gender.append(0)
            else:
                gender.append(1)
            grams.append(" ".join(data[entry].grams))
            posTagVectorizer.append(data[entry].posTagDict)
    i = 0
    tfidf = TfidfVectorizer()
    tfidfGram = tfidf.fit_transform(grams)
    count = CountVectorizer()
    #bowPosTag = count.fit_transform(posTagVectorizer)
    data = {'age':  age,'gender': gender}

    originalDF = pd.DataFrame (data, columns = ['age','gender'])
    #posTagDF = pd.DataFrame(bowPosTag.toarray(),columns=count.get_feature_names())
    tfidfDF = pd.DataFrame(tfidfGram.toarray(),columns=tfidf.get_feature_names())
    originalDF = pd.concat([originalDF,tfidfDF], axis=1, sort=False)
    originalDF.to_csv(r'../data/dataframe.csv', index = False)

import sklearn.utils
from sklearn.preprocessing import StandardScaler
df_female = originalDF.drop(["age","gender"], axis=1)
Clus_dataSet = StandardScaler().fit_transform(df_female)
print(originalDF.head(10))
# Compute DBSCAN
db = DBSCAN(eps=1.0, min_samples=20).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
originalDF['Clus_Db']=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

# A sample of clusters
print(originalDF[['age','gender','Clus_Db']].head())

# number of labels
print("number of labels: ", set(labels))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = Clus_dataSet[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = Clus_dataSet[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % realClusterNum)
plt.show()

n_noise_ = list(labels).count(-1)
print('number of noise(s): ', n_noise_)

#Visualization
for clust_number in set(labels):
    clust_set = originalDF[originalDF.Clus_Db == clust_number]
    if clust_number != -1:
        print ("Cluster "+str(clust_number)+', Avg Age: '+ str(round(np.mean(clust_set.age)))+\
               ', Avg Gender: '+ str(round(np.mean(clust_set['gender'])))+', Count: '+ str(np.count_nonzero(clust_set.index)))
sound()