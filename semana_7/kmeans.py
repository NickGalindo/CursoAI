import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans


# TODO: define data in class

convergence = 100
loss,  center_points, center_arrays = [], [], []
centers = np.random.uniform((np.min(data), np.max(data)), size=(3, 2)) # Hyperparam k is defined here

while(convergence > 0.01):
    center_array = np.zeros(len(data))

    for i, x in enumerate(data):
        closest_center = np.argmin([np.linalg.norm(x - center) for center in centers]) 
        center_array[i] = closest_center

    new_centers = np.array([np.mean(data[center_array == i], axis=0) for i in range(num_centers)])    
    convergence = np.sum(np.abs(new_centers - centers))
    centers = new_centers

    center_points.append(new_centers)
    center_arrays.append(center_array)

plt.scatter([c[0] for c in centers], [c[1] for c in centers], s="5", c="r")
plt.scatter([d[0] for d in data], [d[1] for d in data], s="1")
plt.show()


# With sklearn making it ezpz
model = KMeans(n_clusters=3)
model.fit(data)

labels = model.predict(data)

pprint(labels)
