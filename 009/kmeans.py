import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Fruit Data (Apple, Banana, Melon)
raw_data = {
    'weight': [1, 5, 2, 3, 10, 11, 15, 9, 100, 120, 115, 116],
    'sugar': [100, 200, 262, 245, 500, 520, 595, 540, 1000, 1200, 1120, 1400]
}

df = pd.DataFrame(raw_data)

# Amount of clusters (k)
k = 3

# Initialise the model
model = KMeans(n_clusters=k)
model.fit(df)

# Array of cluster number
labels = model.labels_
print('Labels: ', labels)

# Array size k with coordinates for centroids
centroids = model.cluster_centers_

# Colours to be used in the plot
colours = ['red', 'green', 'blue', 'yellow']

# Plots all the points
y = 0
for x in labels:
    plt.scatter(df.iloc[y, 0], df.iloc[y, 1], color=colours[x])
    y += 1

# Plots the centroids
for x in range(k):
    crosses = plt.plot(centroids[x, 0], centroids[x, 1], 'kx')
    plt.setp(crosses, ms=10.0, mew=3.0)

# Show the plot
plt.show()
