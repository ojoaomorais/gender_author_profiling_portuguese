from sklearn.decomposition import PCA

print(__doc__)

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

# #############################################################################
# Generate sample data
data = m.getData()
id = []
age = []
gender = []
textCount = []
for entry in data.keys():
    if data[entry].age != "":
        id.append(int(entry))
        age.append(int(data[entry].age))
        if data[entry].gender == "male":
            gender.append(0)
        else:
            gender.append(1)
        textCount.append(data[entry].adjCount)
json = {"age":age,"gender":gender, "adjCount":textCount}

df = pd.DataFrame (json, columns = ["age","adjCount","gender"])

df = df.values.astype("float32", copy = False)

stscaler = StandardScaler().fit(df)
df = stscaler.transform(df)

dbsc = DBSCAN(eps = .7, min_samples = 3).fit(df)

core_samples = dbsc.core_sample_indices_
labels = dbsc.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# Plot result
import pylab as pl
from itertools import cycle

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Black removed and is used for noise instead.
colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in np.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    x = []
    y = []
    z = []
    for index in class_members:
        x.append(age[index])
        y.append(gender[index])
        z.append(textCount[index])
        if index in core_samples and k != -1:
            markersize = 14
        else:
            markersize = 6

ax.scatter(x, y, z)
pl.title('Estimated number of clusters: %d' % n_clusters_)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
