from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD


def sound():
    from beepy import beep
    beep(sound=1)  # integer as argument

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pylab import *
import main as m
from clusteval import clusteval
from sklearn.feature_extraction.text import CountVectorizer

data = m.getReviewData()
id = []
idade = []
gender = []
grams = []
posTagVectorizer = []
for entry in data.keys():
    if data[entry].idade != "":
        id.append(int(entry))
        idade.append(int(data[entry].idade))
        gender.append(data[entry].gender)
        grams.append(" ".join(data[entry].grams))
        #posTagVectorizer.append(data[entry].posTagDict)
i = 0
vectorizer = TfidfVectorizer(max_df=0.55,min_df=27)
X = vectorizer.fit_transform(grams)
data = {'idade': idade,'gender': gender}

originalDF = pd.DataFrame (data, columns = ['idade','gender'])
print("Antes do Resample")
print(originalDF['idade'].value_counts())
print(originalDF['gender'].value_counts())

tfidfDF = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
originalDF = pd.concat([originalDF,tfidfDF], axis=1, sort=False)

df_minority  = originalDF[originalDF['idade']==0]
df_ok = originalDF[originalDF['idade']==1]
df_majority = originalDF[originalDF['idade']==2]

### Now, downsamples majority labels equal to the number of samples in the minority class
df_majority = df_majority.sample(len(df_minority), random_state=0)
df_ok = df_ok.sample(len(df_minority), random_state=0)
### concat the majority and minority dataframes
df = pd.concat([df_majority,df_minority,df_ok])
## Shuffle the dataset to prevent the model from getting biased by similar samples
originalDF = df.sample(frac=1, random_state=0)

print("Pós do Resample")
print(originalDF['idade'].value_counts())
print(originalDF['gender'].value_counts())

newDF = originalDF.drop('idade',axis=1)

# print("Performing LSA Dimension Reduction")
svd = TruncatedSVD(n_components=200)
newDF = svd.fit_transform(newDF.values)
#
# print("AFTER LSA matrix reduction: ")
# print("n_samples: %d, n_features: %d" % X.shape)

# # # #K-MEANS CLUSTERING
# km = KMeans(n_clusters=6)
# print("Clustering sparse data with %s" % km)
# # t0 = time()
# km.fit(newDF)
# y_means = km.predict(newDF)

# dbscan = DBSCAN()
# y_means = dbscan.fit_predict(newDF)

# originalDF["clusters"] = y_means

ce = clusteval(method='silhouette')
ce.fit(newDF)
ce.plot()
ce.dendrogram()
ce.scatter(newDF)

groupedDF = originalDF[["idade","gender","clusters"]].groupby("clusters")
for name,group in groupedDF:
    cluster = name
    maleCount = 0
    femaleCount = 0
    less20 = 0
    less30 = 0
    larger30 = 0
    for row in group.values:
        if row[2] == cluster:
            if row[0] != "":
                if int(row[0]) == 0:
                    less20 += 1
                elif int(row[0]) == 1:
                    less30 += 1
                else:
                    larger30 += 1
            if row[1] == 0:
                maleCount += 1
            else:
                femaleCount += 1
    print("========== Cluster %s===========" % cluster)
    print("Female Count : %s" % femaleCount)
    print("Male Count : %s" % maleCount)
    print("< 20 : %s" % less20)
    print("> 20 and < 30 : %s" % less30)
    print("> 30 : %s" % larger30)
    print("=============================")
    cluster += 1


# ce = clusteval(method='silhouette')
# ce.fit(newDF.values)
# ce.plot()
# ce.dendrogram()
# ce.scatter(newDF.values)

# #PLOTS K-MEANS CLUSTERING
#
# plt.scatter(newDF.values[:,0],newDF.values[:,1], c=y_means, s=50, cmap='viridis')
# plt.show()