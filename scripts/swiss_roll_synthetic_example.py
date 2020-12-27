import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import coordinate_chart
import particle_filter

# Parameters
train_resolution = 0.2
target_dim = 2
neighbors_k = 5
n_particles = 200
exploration_factor = 0.1
pos_var = 0.005

# Create the swiss roll dataset
# Note that we're not using random sampling, but rather evenly distributing points along the manifold
s = np.arange(0, 1, train_resolution)
t = np.arange(2 * np.pi, 6 * np.pi, train_resolution)
s_len = len(s)
t_len = len(t)
s = np.repeat(s, t_len)
t = np.tile(t, s_len)
data = np.array([0.05 * t * np.cos(t), s, 0.05 * t * np.sin(t)]).transpose()

ground_truth = np.array([0.05 * 4 * np.pi, 0.5, 0.0])

# Coordinate chart setup
cc = coordinate_chart.CoordinateChart(data, target_dim, neighbors_k)

# Particle filter setup
def likelihood(point):
	source_dim_point = cc.inverse_mapping([point])[0]
	return 1 / (1 + np.linalg.norm(source_dim_point - ground_truth))

def diffuser(point):
	return point + np.random.multivariate_normal(np.zeros(target_dim), pos_var*np.eye(target_dim))

pf = particle_filter.ParticleFilter(target_dim, n_particles, exploration_factor, True, cc.uniform_sample, likelihood, diffuser)

while True:
	pf.weight()
	mle = pf.predict_mle()
	mean = pf.predict_mean()

	manifold_particles = 




		pf.weight()
		mle = pf.predict_mle()
		mean = pf.predict_mean()
		plt.scatter(pf.particles[:,0], pf.particles[:,1], c=pf.weights)
		plt.scatter([mle[0]], [mle[1]], marker="D", color="black")
		plt.scatter([mean[0]], [mean[1]], marker="X", color="black")
		plt.scatter([0], [0], marker="*", color="red")
		plt.xlim(-1, 1)
		plt.ylim(-1, 1)
		plt.show()
		pf.resample()
		pf.diffuse()
