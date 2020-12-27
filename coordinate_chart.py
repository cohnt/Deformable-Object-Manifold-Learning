import numpy as np
from sklearn.manifold import Isomap
from scipy.spatial import Delaunay

class CoordinateChart():
	def __init__(self, train_data, target_dim, neighbors_k):
		self.train_data = train_data
		self.source_dim = train_data.shape[1]
		self.target_dim = target_dim
		self.neighbors_k = neighbors_k

		self.ism = Isomap(n_neighbors=self.neighbors_k, n_components=self.target_dim, n_jobs=-1)
		self.embedding = self.ism.fit_transform(train_data)
		self.tri = Delaunay(self.embedding, qhull_options="QJ")

		self.mins = np.min(self.embedding, axis=0)
		self.maxs = np.max(self.embedding, axis=0)
		self.p2p = self.maxs - self.mins

	def inverse_mapping(self, points):
		# points should be a numpy array of shape [*,target_dim]
		# This function will return a numpy array of shape [*,source_dim]
		mapped_points = np.zeros(len(points), self.source_dim)
		for i in range(len(points)):
			# Simplex lookup
			point = points[i]
			simplex_num = self.tri.find_simplex(point)
			if simplex_num == -1:
				print "Error: coordinate outside of convex hull!"
				print point
				raise ValueError
			simplex_indices = self.tri.simplices[simplex_num]
			simplex = self.tri.points[simplex_indices]

			# Write as convex combination of simplex vertices
			A = np.vstack((simplex.T, np.ones((1, self.target_dim+1))))
			b = np.vstack((point.reshape(-1, 1), np.ones((1, 1))))
			convex_comb = np.linalg.solve(A, b)
			convex_comb = np.asarray(convex_comb).flatten()

			# Interpolate to the higher dimensional space
			factors = np.zeros(self.train_data.length)
			factors[simplex_indices] = convex_comb
			factors_mat = np.diag(factors)
			mapped_point = np.sum(np.matmul(factors_mat, self.train_data), axis=0).flatten()
			mapped_points[i,:] = mapped_point
		return mapped_points

	def check_domain(self, points):
		simplex_nums = self.tri.find_simplex(points)
		return simplex_nums != -1

	def uniform_sample(self, n):
		start = np.random.rand(n,target_dim)
		return np.matmul(start, np.diag(self.p2p)) + np.mins