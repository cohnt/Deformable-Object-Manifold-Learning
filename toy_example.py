import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, np.pi, 100)
data = np.array([0.5 + (0.5 * np.cos(t)), 0.5 * np.sin(t)]).transpose()

actual = np.array([0.5, 0.5])

from sklearn.manifold import Isomap
ism = Isomap(n_neighbors=5, n_components=1)
embedding = ism.fit_transform(data)

from scipy.stats import multivariate_normal
def likelihood(point):
	return multivariate_normal.pdf(point, mean=actual, cov=np.eye(2))

######################
# 2D Particle Filter #
######################

class SimpleParticle():
	def __init__(self, xy=None):
		if xy is None:
			self.xy = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)])
		else:
			self.xy = xy

		self.raw_weight = None
		self.normalized_weight = None

num_particles = 100
exploration_factor = 0
particles = [SimpleParticle() for i in range(num_particles)]
iter_num = 0
xy_var = 0.1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.draw()
plt.pause(0.1)

while True:
	iter_num = iter_num + 1

	# Compute weights
	normalization_factor = 0
	for p in particles:
		p.raw_weight = likelihood(p.xy)
		normalization_factor = normalization_factor + p.raw_weight
	for p in particles:
		p.normalized_weight = p.raw_weight / normalization_factor

	# Predict
	normalized_weights = [p.normalized_weight for p in particles]
	mle = particles[np.argmax(normalized_weights)]
	average = np.average([p.xy for p in particles], axis=0, weights=normalized_weights)

	# Display
	ax.clear()
	coords = np.array([p.xy for p in particles])
	weights = np.array([p.raw_weight for p in particles])
	ax.scatter(coords[:,0], coords[:,1], cmap=plt.cm.cool, c=weights)
	ax.scatter([mle.xy[0]], [mle.xy[1]], color="black", marker="*")
	ax.scatter([average[0]], [average[1]], color="black", marker="x")
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
		newParticles.append(particles[chkIdx])
	for i in range(len(newParticles), num_particles):
		newParticles.append(SimpleParticle())
	particles = newParticles

	# Diffusion Noise
	for p in particles:
		p.xy = p.xy + np.random.multivariate_normal(np.zeros(2), xy_var*np.eye(2))