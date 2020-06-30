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
			self.xyz = np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1), np.random.uniform(-1, 1)])
		else:
			self.xyz = xyz

		self.raw_weight = None
		self.normalized_weight = None

num_particles = 100
exploration_factor = 0
particles = [SimpleParticle() for i in range(num_particles)]
iter_num = 0
xyz_var = 0.005
convergence_threshold = 0.005

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-1, 1)
ax.set_ylim(0, 1)
ax.set_zlim(-1, 1)
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
	mle = particles[np.argmax(normalized_weights)]
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
	ax.scatter(coords[:,0], coords[:,1], cmap=plt.cm.cool, c=weights)
	ax.scatter([mle.xyz[0]], [mle.xyz[1]], color="black", marker="*")
	ax.scatter([average[0]], [average[1]], color="black", marker="x")
	ax.scatter([actual[0]], [actual[1]], color="green", marker="+")
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
		p.xyz = p.xyz + np.random.multivariate_normal(np.zeros(len(actual)), xyz_var*np.eye(len(actual)))

print "Original Particle Filter Results:"
print "Number of iterations:", (iter_num - 1)
print "Final prediction:", prediction
print "Error:", np.linalg.norm(prediction - actual)