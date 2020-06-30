import numpy as np

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
			self.xy = np.array([np.random.uniform(0, 1), np.random.uniform(0, 0.5)])
		else:
			self.xy = xy

		self.raw_weight = None
		self.normalized_weight = None

num_particles = 100
exploration_factor = 0
particles = [Particle() for i in range(num_particles)]
iter_num = 0

while True:
	iter_num = iter_num + 1

	# Compute raw weights
	normalization_factor = 0
	for p in particles:
		p.raw_weight = likelihood(p.xy)
		normalization_factor = normalization_factor + p.raw_weight
	for p in particles:
		p.normalized_weight = p.raw_weight / normalization_factor