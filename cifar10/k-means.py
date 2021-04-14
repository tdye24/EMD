import numpy as np
from collections import defaultdict
from random import uniform
from math import sqrt
from sklearn.cluster import KMeans

data = np.load('./newX.npy')
print(data.shape)

kmeans = KMeans(n_clusters=5, max_iter=10000)
kmeans = kmeans.fit(data)
labels = kmeans.predict(data)

print(labels)
np.save('./clusters', labels)