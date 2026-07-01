import random
# iris dataset from sklearn
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import numpy as np

data = load_iris()['data']
data = data/np.linalg.norm(data, axis=1).reshape(-1, 1)
data = [list(point) for point in list(data)]
targets = ["" for _ in range(len(data))]
num_clusters = 3
num_points = len(data)
num_features = len(data[0])

# randomly select 3 points as the initial centroids
centroids = random.sample(list(data), num_clusters)
for i in range(20):
    closest_centroids={}
    for j , datum in enumerate(data):
        dist = []
        for centroid in centroids:
            distance = sum([(a-b)**2 for a,b in zip(datum, centroid)])
            dist.append(distance)
        closest_centroid = dist.index(min(dist))
        closest_centroids[closest_centroid] = closest_centroids.get(closest_centroid, []) + [datum]
        targets[j] = closest_centroid

    for x in  closest_centroids:
        new_centroid = ([sum(i)/len(closest_centroids[x]) for i in zip(*closest_centroids[x])])
        centroids[x] = new_centroid

print(centroids)
print(targets)
print(closest_centroids)

#marker style for each cluster

plt.scatter([i[2] for i in data], [i[3] for i in data], c=targets, marker='x')
plt.scatter([i[2] for i in centroids], [i[3] for i in centroids], c='r')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
pass
