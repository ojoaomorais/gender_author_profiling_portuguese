from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

def sound():
    from beepy import beep
    beep(sound=1)  # integer as argument

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pylab import *
import main as m
import datetime as dt
from clusteval import clusteval
from sklearn.feature_extraction.text import CountVectorizer

data = m.getData()
id = []
idade = []
gender = []
predictedGender = []
for entry in data.keys():
    if data[entry].idade != "":
        id.append(int(entry))
        idade.append(int(data[entry].idade))
        gender.append(data[entry].gender)
        predictedGender.append(data[entry].predictedGender)
        #grams.append(" ".join(data[entry].grams))
        #cC.append(data[entry].cC)
        #bS.append(data[entry].bS)
        #pC.append(data[entry].pC)
        #posTagVectorizer.append(data[entry].posTagDict)
        #grams2.append(" ".join(data[entry].grams2))
i = 0
vectorizer = TfidfVectorizer(min_df=0.06)
#X = vectorizer.fit_transform(grams)
#vectorizer2 = TfidfVectorizer()
#Y = vectorizer2.fit_transform(grams2)

data = {'idade': idade,'gender': gender,"predictedGender":predictedGender}
corretsMasculino = 0
corretsFeminino = 0
wrongsMasculino = 0
wrongsFeminino = 0

originalDF = pd.DataFrame (data, columns = ['idade','gender','predictedGender'])

for index, row in originalDF.iterrows():
    if row.gender:
        if row.gender == 0:
            if row.predictedGender == "Fem":
                wrongsMasculino += 1
            elif row.predictedGender == "Masc":
                corretsMasculino += 1
        if row.gender == 1:
            if row.predictedGender == "Fem":
                corretsFeminino += 1
            elif row.predictedGender == "Masc":
                wrongsFeminino += 1
print("NUMERO DE CORRETOS")
print("Feminino")
print(corretsFeminino)
print("Masculino")
print(corretsMasculino)
print("NUMERO DE INCORRETOS")
print("Feminino")
print(wrongsFeminino)
print("Masculino")
print(wrongsMasculino)
print("Antes do Resample")
print(originalDF['idade'].value_counts())
print(originalDF['gender'].value_counts())

#tfidfDF = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
#tfidfDF2 = pd.DataFrame(Y.toarray(),columns=vectorizer2.get_feature_names())
#bowDF = pd.DataFrame(Y.toarray(),columns=cvect.get_feature_names())
# originalDF = pd.concat([originalDF,tfidfDF], axis=1, sort=False)
#
# df_majority  = originalDF[originalDF['idade']==0]
# df_ok = originalDF[originalDF['idade']==1]
# df_minority = originalDF[originalDF['idade']==2]
#
# # ### Now, downsamples majority labels equal to the number of samples in the minority class
# df_majority = df_majority.sample(15000, random_state=1)
# df_ok = df_ok.sample(15000, random_state=1)
# df_minority = df_minority.sample(15000, random_state=1)
# # ### concat the majority and minority dataframes
# df = pd.concat([df_majority,df_minority,df_ok])
# # ## Shuffle the dataset to prevent the model from getting biased by similar samples
# originalDF = df.sample(frac=1, random_state=0)
#
# print("Pós do Resample")
# print(originalDF['idade'].value_counts())
# print(originalDF['gender'].value_counts())
#
# newDF = originalDF.drop(['idade',"gender"],axis=1)
#
# # print("Performing LSA Dimension Reduction")
# # svd = TruncatedSVD(n_components=100)
# # newDF = svd.fit_transform(newDF.values)
# #
# # print("AFTER LSA matrix reduction: ")
# print("n_samples: %d, n_features: %d" % newDF.shape)
# #
# # print("Iniciou o Fit")
# # neigh = NearestNeighbors(n_neighbors=5)
# # nbrs = neigh.fit(newDF.values)
# # print("Finalizou o Fit")
# # distances, indices = nbrs.kneighbors(newDF.values)
# # print("Finalizou Distances")
# #
# # distances = np.sort(distances, axis=0)
# # distances = distances[:, 1]
# # plt.plot(distances)
# # plt.show()
#
# # # # # #K-MEANS CLUSTERING
# km = KMeans(n_clusters=6,n_jobs=-1)
# print("Clustering sparse data with %s" % km)
# # t0 = time()
# km.fit(newDF)
# y_means = km.predict(newDF)
#
# # dbscan = DBSCAN(eps=0.3,min_samples=100,n_jobs=-1,metric="cosine").fit(newDF)
# # y_means = dbscan.labels_
# originalDF["clusters"] = y_means
#
# groupedDF = originalDF[["idade","gender","clusters"]].groupby("clusters")
# import seaborn as sns;
#
#
# for name,group in groupedDF:
#     cluster = name
#     femaleCount = 0
#     femaleLess20 = 0
#     femaleLess30 = 0
#     femaleLarger30 = 0
#     maleLess20 = 0
#     maleLess30 = 0
#     maleLess40 = 0
#     for row in group.values:
#         if row[2] == cluster:
#             if row[1] == 0:
#                 if row[0] != "":
#                     if int(row[0]) == 0:
#                         maleLess20 += 1
#                     elif int(row[0]) == 1:
#                         maleLess30 += 1
#                     else:
#                         maleLess40 += 1
#             else:
#                 if row[0] != "":
#                     if int(row[0]) == 0:
#                         femaleLess20 += 1
#                     elif int(row[0]) == 1:
#                         femaleLess30 += 1
#                     else:
#                         femaleLarger30 += 1
#     print("========== Cluster %s===========" % cluster)
#
#     print("Female < 20 : %s" % femaleLess20)
#     print("Female > 20 and < 30 : %s" % femaleLess30)
#     print("Female > 30 : %s" % femaleLarger30)
#
#     print("Male < 20 : %s" % maleLess20)
#     print("Male > 20 and < 30 : %s" % maleLess30)
#     print("Male > 30 : %s" % maleLess40)
#     print("=============================")
#     cluster += 1
#
