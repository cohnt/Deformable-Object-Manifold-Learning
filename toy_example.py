import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

s = np.arange(0, 1, 0.05)
t = np.arange(2 * np.pi, 6 * np.pi, 0.05)
s_len = len(s)
t_len = len(t)
s = np.repeat(s, t_len)
t = np.tile(t, s_len)
data = np.array([0.05 * t * np.cos(t), s, 0.05 * t * np.sin(t)]).transpose()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(data[:,0], data[:,1], data[:,2])
# plt.show()

x_min = -1
x_max = 1
y_min = 0
y_max = 1
z_min = -1
z_max = 1

actual = np.array([0.05 * 4 * np.pi, 0.5, 0])

from scipy.stats import multivariate_normal
def likelihood(point):
	return multivariate_normal.pdf(point, mean=actual, cov=np.eye(len(actual)))

######################
# 2D Particle Filter #
######################

class SimpleParticle():
	def __init__(self, xyz=None):
		if xyz is None:
			self.xyz = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), np.random.uniform(z_min, z_max)])
		else:
			self.xyz = xyz

		self.raw_weight = None
		self.normalized_weight = None

num_particles = 100
exploration_factor = 0
pos_var = 0.005
convergence_threshold = 0.005
particles = [SimpleParticle() for i in range(num_particles)]
iter_num = 0

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
plt.draw()
plt.pause(0.1)

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
		change = np.linalg.norm(average - prediction)
		prediction = average
		if change < convergence_threshold:
			break

	print "Iteraton %d: predicted" % iter_num, prediction

	# Display
	ax.clear()
	ax.set_xlim(-1, 1)
	ax.set_ylim(0, 1)
	ax.set_zlim(-1, 1)
	coords = np.array([p.xyz for p in particles])
	weights = np.array([p.raw_weight for p in particles])
	ax.scatter(coords[:,0], coords[:,1], coords[:,2], cmap=plt.cm.cool, c=weights)
	ax.scatter([mle[0]], [mle[1]], [mle[2]], color="black", marker="*")
	ax.scatter([average[0]], [average[1]], [average[2]], color="black", marker="x")
	ax.scatter([actual[0]], [actual[1]], [actual[2]], color="green", marker="+")
	plt.draw()
	plt.pause(0.1)

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

print "Original Particle Filter Results:"
print "Number of iterations:", (iter_num - 1)
print "Final prediction:", prediction
print "Error:", np.linalg.norm(prediction - actual)

##########################
# Isomap Particle Filter #
##########################

from sklearn.manifold import Isomap
ism = Isomap(n_neighbors=5, n_components=2)
embedding = ism.fit_transform(data)

from scipy.spatial import Delaunay
interpolator = Delaunay(embedding, qhull_options="QJ")

def compute_interpolation(interpolator, embedding_coords):
	simplex_num = interpolator.find_simplex(embedding_coords)
	if simplex_num != -1:
		simplex_indices = interpolator.simplices[simplex_num]
		simplex = interpolator.points[simplex_indices]

		# Compute barycentric coordinates
		A = np.vstack((simplex.T, np.ones((1, 3))))
		b = np.vstack((embedding_coords.reshape(-1, 1), np.ones((1, 1))))
		b_coords = np.linalg.solve(A, b)
		b = np.asarray(b_coords).flatten()

		# Interpolate back to the manifold
		mult_vec = np.zeros(len(data))
		mult_vec[simplex_indices] = b
		curve = np.sum(np.matmul(np.diag(mult_vec), data), axis=0).reshape(-1,3)
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

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
plt.draw()
plt.pause(0.1)

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
		change = np.linalg.norm(average - prediction)
		prediction = average
		if change < convergence_threshold:
			break

	print "Iteraton %d: predicted" % iter_num, prediction

	# Display
	ax.clear()
	ax.set_xlim(-1, 1)
	ax.set_ylim(0, 1)
	ax.set_zlim(-1, 1)
	coords = np.array([p.point for p in particles])
	weights = np.array([p.raw_weight for p in particles])
	ax.scatter(coords[:,0], coords[:,1], coords[:,2], cmap=plt.cm.cool, c=weights)
	ax.scatter([mle[0]], [mle[1]], [mle[2]], color="black", marker="*")
	ax.scatter([average[0]], [average[1]], [average[2]], color="black", marker="x")
	ax.scatter([actual[0]], [actual[1]], [actual[2]], color="green", marker="+")
	plt.draw()
	plt.pause(0.1)

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

print "ISOMAP Particle Filter Results:"
print "Number of iterations:", (iter_num - 1)
print "Final prediction:", prediction
print "Error:", np.linalg.norm(prediction - actual)