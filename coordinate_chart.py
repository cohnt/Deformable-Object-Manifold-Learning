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
		if self.target_dim > 1:
			self.tri = Delaunay(self.embedding, qhull_options="QJ")
		else:
			self.tri = Delaunay1D(self.embedding)

		self.mins = np.min(self.embedding, axis=0)
		self.maxs = np.max(self.embedding, axis=0)
		self.p2p = self.maxs - self.mins

	def inverse_mapping(self, points):
		# points should be a numpy array of shape [target_dim] or [*,target_dim]
		# This function will return a numpy array of shape [source_dim] or [*,source_dim]
		if points.ndim == 1:
			return self.single_inverse_mapping(points)
		elif points.ndim == 2:
			mapped_points = np.zeros((len(points), self.source_dim))
			for i in range(len(points)):
				mapped_points[i,:] = self.single_inverse_mapping(points[i])
			return mapped_points

	def single_inverse_mapping(self, point):
		# Simplex lookup
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
		factors = np.zeros(len(self.train_data))
		factors[simplex_indices] = convex_comb
		factors_mat = np.diag(factors)
		mapped_point = np.sum(np.matmul(factors_mat, self.train_data), axis=0).flatten()

		return mapped_point

	def check_domain(self, points):
		if points.ndim == 1:
			return (self.tri.find_simplex(points) != -1)
		elif points.ndim == 2:
			simplex_nums = self.tri.find_simplex(points)
			return simplex_nums != -1

	def uniform_sample(self, n):
		points = np.zeros((n,self.target_dim))
		for i in range(n):
			while True:
				start = np.random.rand(self.target_dim)
				point = np.matmul(start, np.diag(self.p2p)) + self.mins
				if self.check_domain(point):
					points[i] = point
					break
		return points

class Delaunay1D():
	def __init__(self, data):
		self.points = data
		self.points_flat = self.points.flatten()
		self.inds = np.argsort(self.points_flat)
		self.sorted_points = self.points_flat[self.inds]
		self.simplices = np.array([self.inds[:-1], self.inds[1:]]).T

	def find_simplex(self, point):
		idx = np.searchsorted(self.sorted_points, point)
		if idx == 0 or idx == len(self.points):
			return -1
		simplex_idx = self.inds[idx][0]
		return simplex_idx

def test_coordinate_chart():
	n = 200
	higher_dim = 3
	source_dim = 2
	k = 8
	initial_data = np.random.rand(n,source_dim)

	from scipy.stats import special_ortho_group
	random_transform = special_ortho_group.rvs(higher_dim)[:,:source_dim]
	train_data = np.matmul(random_transform, initial_data.T).T

	cc = CoordinateChart(train_data, source_dim, k)

	# Embedding plot
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(1,2)
	axes[0].scatter(initial_data[:,0], initial_data[:,1], c=initial_data[:,0])
	axes[1].scatter(cc.embedding[:,0], cc.embedding[:,1], c=initial_data[:,0])
	plt.show()

	# Check inverse mapping
	import mpl_toolkits.mplot3d.axes3d
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 2, 1, projection="3d")
	ax2 = fig.add_subplot(1, 2, 2, projection="3d")
	ax1.scatter(train_data[:,0], train_data[:,1], train_data[:,2], c=initial_data[:,0]);
	transformed_data = cc.inverse_mapping(cc.embedding)
	ax2.scatter(transformed_data[:,0], transformed_data[:,1], transformed_data[:,2], c=initial_data[:,0])
	plt.show()

	# Check random sampling
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(cc.embedding[:,0], cc.embedding[:,1], color="red")
	rand_sample = cc.uniform_sample(100)
	ax.scatter(rand_sample[:,0], rand_sample[:,1], color="blue")
	plt.show()

if __name__ == "__main__":
	test_coordinate_chart()
	data = np.array([1, 2, 3, 4, 6, 5])
	data = np.array([data, data]).T
	cc = CoordinateChart(data, 1, 3)
	print cc.embedding
	print cc.single_inverse_mapping(np.array([0]))