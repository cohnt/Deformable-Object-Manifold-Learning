import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.stats import multivariate_normal
from sklearn.manifold import Isomap
from scipy.spatial import Delaunay

max_iters = 1000
norm = np.inf
train_resolution = 0.2
num_particles = 200
exploration_factor = 0.1
pos_var = 0.005
convergence_threshold = 0.005

dim_list = range(0, 7+1)
num_runs = 25
orig_data = np.zeros((len(dim_list), num_runs))
my_data = np.zeros((len(dim_list), num_runs))

# Graph number of iterations for convergence by added dimension.
for extra_dims in dim_list:
	for run_num in range(num_runs):
		print "%d extra dimensions. Run #%d..." % (extra_dims, run_num)

		s = np.arange(0, 1, train_resolution)
		t = np.arange(2 * np.pi, 6 * np.pi, train_resolution)
		s_len = len(s)
		t_len = len(t)
		s = np.repeat(s, t_len)
		t = np.tile(t, s_len)
		data = np.array([0.05 * t * np.cos(t), s, 0.05 * t * np.sin(t)]).transpose()
		data = np.array([np.append(d, np.zeros(extra_dims)) for d in data])

		x_min = -1
		x_max = 1
		y_min = -1
		y_max = 1
		z_min = -1
		z_max = 1

		actual = np.array([0.05 * 4 * np.pi, 0.5, 0.0])
		actual = np.append(actual, np.zeros(extra_dims))

		def likelihood(point):
			return multivariate_normal.pdf(point, mean=actual, cov=0.5*np.eye(len(actual)))

		######################
		# 2D Particle Filter #
		######################

		class SimpleParticle():
			def __init__(self, xyz=None):
				if xyz is None:
					self.xyz = np.random.uniform(-1, 1, size=len(actual))
				else:
					self.xyz = xyz

				self.raw_weight = None
				self.normalized_weight = None

		particles = [SimpleParticle() for i in range(num_particles)]
		iter_num = 0
		prediction = None
		while True:
			iter_num = iter_num + 1

			# Compute weights
			normalization_factor = 0
			for p in particles:
				p.raw_weight = likelihood(p.xyz)
				normalization_factor = normalization_factor + p.raw_weight
			for p in particles:
				p.normalized_weight = p.raw_weight / normalization_factor

			# Predict
			normalized_weights = [p.normalized_weight for p in particles]
			mle = particles[np.argmax(normalized_weights)].xyz
			average = np.average([p.xyz for p in particles], axis=0, weights=normalized_weights)

			if prediction is None:
				prediction = average
			else:
				change = np.linalg.norm(average - prediction, norm)
				prediction = average
				if change < convergence_threshold:
					break
				if iter_num >= max_iters:
					break

			# Resample
			newParticles = []
			cs = np.cumsum([normalized_weights])
			step = 1/float((num_particles * (1-exploration_factor))+1)
			chkVal = step
			chkIdx = 0
			for i in range(0, int(np.ceil(num_particles * (1-exploration_factor)))):
				while cs[chkIdx] < chkVal:
					chkIdx = chkIdx + 1
				chkVal = chkVal + step
				newParticles.append(SimpleParticle(xyz=particles[chkIdx].xyz))
			for i in range(len(newParticles), num_particles):
				newParticles.append(SimpleParticle())
			particles = newParticles

			# Diffusion Noise
			for p in particles:
				p.xyz = p.xyz + np.random.multivariate_normal(np.zeros(len(actual)), pos_var*np.eye(len(actual)))

		orig_data[extra_dims, run_num] = iter_num
		print "Ordinary particle filter required %d iterations" % iter_num

		##########################
		# Isomap Particle Filter #
		##########################

		ism = Isomap(n_neighbors=5, n_components=2)
		embedding = ism.fit_transform(data)

		interpolator = Delaunay(embedding, qhull_options="QJ")

		def compute_interpolation(interpolator, embedding_coords):
			simplex_num = interpolator.find_simplex(embedding_coords)
			if simplex_num != -1:
				simplex_indices = interpolator.simplices[simplex_num]
				simplex = interpolator.points[simplex_indices]

				# Compute barycentric coordinates
				A = np.vstack((simplex.T, np.ones((1, 2+1))))
				b = np.vstack((embedding_coords.reshape(-1, 1), np.ones((1, 1))))
				b_coords = np.linalg.solve(A, b)
				b = np.asarray(b_coords).flatten()

				# Interpolate back to the manifold
				mult_vec = np.zeros(len(data))
				mult_vec[simplex_indices] = b
				curve = np.sum(np.matmul(np.diag(mult_vec), data), axis=0).reshape(-1,len(actual))
				return curve[0]
			else:
				print "Error: outside of convex hull!"
				raise ValueError

		class EmbeddingParticle():
			def __init__(self, pos=None):
				if pos is None:
					ind = np.random.randint(0, len(embedding))
					self.pos = embedding[ind]
				else:
					self.pos = pos

				self.compute_point()

				self.raw_weight = None
				self.normalized_weight = None

			def compute_point(self):
				self.point = compute_interpolation(interpolator, self.pos)

		particles = [EmbeddingParticle() for i in range(num_particles)]
		iter_num = 0
		prediction = None
		while True:
			iter_num = iter_num + 1

			# Compute weights
			normalization_factor = 0
			for p in particles:
				p.raw_weight = likelihood(p.point)
				normalization_factor = normalization_factor + p.raw_weight
			for p in particles:
				p.normalized_weight = p.raw_weight / normalization_factor

			# Predict
			normalized_weights = [p.normalized_weight for p in particles]
			mle = particles[np.argmax(normalized_weights)].point
			average = np.average([p.point for p in particles], axis=0, weights=normalized_weights)

			if prediction is None:
				prediction = average
			else:
				change = np.linalg.norm(average - prediction, norm)
				prediction = average
				if change < convergence_threshold:
					break
				if iter_num >= max_iters:
					break

			# Resample
			newParticles = []
			cs = np.cumsum([normalized_weights])
			step = 1/float((num_particles * (1-exploration_factor))+1)
			chkVal = step
			chkIdx = 0
			for i in range(0, int(np.ceil(num_particles * (1-exploration_factor)))):
				while cs[chkIdx] < chkVal:
					chkIdx = chkIdx + 1
				chkVal = chkVal + step
				newParticles.append(EmbeddingParticle(pos=particles[chkIdx].pos))
			for i in range(len(newParticles), num_particles):
				newParticles.append(EmbeddingParticle())
			particles = newParticles

			# Diffusion Noise
			for p in particles:
				while True:
					noise = np.random.multivariate_normal(np.zeros(len(p.pos)), pos_var*np.eye(len(p.pos)))
					if interpolator.find_simplex(p.pos + noise) != -1:
						p.pos = p.pos + noise
						break
				p.compute_point()

		my_data[extra_dims, run_num] = iter_num
		print "Coorinate chart particle filter required %d iterations" % iter_num

print "\n"
print "Raw data:"
print orig_data
print my_data

orig_means = np.mean(orig_data, axis=1)
my_means = np.mean(my_data, axis=1)

print "\n"
print "Means:"
print orig_means
print my_means

orig_err_bars = [orig_means - np.min(orig_data, axis=1), np.max(orig_data, axis=1) - orig_means]
my_err_bars = [my_means - np.min(my_data, axis=1), np.max(my_data, axis=1) - my_means]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(dim_list, orig_means, yerr=orig_err_bars, label="Regular PF", capsize=5)
ax.errorbar(dim_list, my_means, yerr=my_err_bars, label="Coordinate Chart PF", capsize=5)
ax.set_xlabel("Data Dimension")
ax.set_ylabel("Average Number of Iterations to Converge")
ax.legend()
plt.show()