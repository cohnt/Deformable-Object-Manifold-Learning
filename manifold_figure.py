import numpy as np
from sklearn import datasets, manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

# Dataset 3D plot

n_points = 250
X, color = datasets.make_s_curve(n_points, random_state=1)
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
	ax.plot(points[[0,1]][:,0], points[[0,1]][:,1], c="black", zorder=1)
	ax.plot(points[[0,2]][:,0], points[[0,2]][:,1], c="black", zorder=1)
	ax.plot(points[[1,2]][:,0], points[[1,2]][:,1], c="black", zorder=1)

ax.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral, s=p_rad_2d**2, zorder=10)
plt.show()

# Triangles in the 3D plot

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for simplex in tri.simplices:
	points = X[simplex]
	x = points[:,0]
	y = points[:,1]
	z = points[:,2]
	verts = [list(zip(x,y,z))]
	ax.add_collection3d(Poly3DCollection(verts))

ax.scatter(X[:,0], X[:,1], X[:,2], c=color, cmap=plt.cm.Spectral, s=p_rad_3d**2)
ax.view_init(4, -72)
plt.show()