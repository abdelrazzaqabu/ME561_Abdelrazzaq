import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#import nothing

# K-Means Clustering Exercise Questions
# Abdelrazzaq Abuhejleh - 20189102002

# Exercise 1: Defining Data Points for Clustering
X = np.array([[12, 39], [20, 36], [28, 30], [18, 52], [29, 54], [33, 46], [24, 55],
              [45, 59], [45, 63], [52, 70], [51, 66], [52, 63], [55, 58], [53, 23],
              [55, 14], [61, 8], [64, 19], [69, 7], [72, 24]])

print('The shape of the array: ', X.shape)

# Exercise 2: Initializing the K-Means Model
Kmeans = KMeans(n_clusters=3)

# Exercise 3: Fitting the Model and Obtaining Labels
Kmeans.fit(X)
print(Kmeans.labels_)

# Exercise 4: Visualizing Clusters
array2 = Kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.scatter(array2[0, 0], array2[0, 1], s=200, c='g', marker='s')
plt.scatter(array2[1, 0], array2[1, 1], s=200, c='r', marker='s')
plt.scatter(array2[2, 0], array2[2, 1], s=200, c='y', marker='s')
plt.show()
