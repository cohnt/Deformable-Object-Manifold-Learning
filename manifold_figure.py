import numpy as np
from sklearn import datasets, manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

# Dataset 3D plot

n_points = 250
X, color = datasets.make_s_curve(n_points, random_state=0)
n_neighbors = 8
n_components = 2

p_rad_3d = 7
p_rad_2d = 10

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:,0], X[:,1], X[:,2], c=color, cmap=plt.cm.Spectral, s=p_rad_3d**2)
ax.view_init(4, -72)
plt.show()

# Embedding plot

Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral, s=p_rad_2d**2)
plt.show()

# Compute Delaunay

tri = Delaunay(Y, qhull_options="QJ")

# Triangles in the embedding plot

fig = plt.figure()
ax = fig.add_subplot(111)

for simplex in tri.simplices:
	points = Y[simplex]
	ax.plot(points[[0,1]][:,0], points[[0,1]][:,1], c="black")
	ax.plot(points[[0,2]][:,0], points[[0,2]][:,1], c="black")
	ax.plot(points[[1,2]][:,0], points[[1,2]][:,1], c="black")

ax.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral, s=p_rad_2d**2)
plt.show()