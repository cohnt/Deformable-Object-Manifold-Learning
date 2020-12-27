import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.stats import multivariate_normal

import coordinate_chart
import particle_filter

# Parameters
train_resolution = 0.15
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
	source_dim_point = cc.single_inverse_mapping(point)
	return multivariate_normal.pdf(source_dim_point, mean=ground_truth, cov=0.5*np.eye(len(ground_truth)))

def diffuser(point):
	while True:
		delta = np.random.multivariate_normal(np.zeros(target_dim), pos_var*np.eye(target_dim))
		if cc.check_domain([point + delta])[0]:
			return point + delta

pf = particle_filter.ParticleFilter(target_dim, n_particles, exploration_factor, True, cc.uniform_sample, likelihood, diffuser)

# Create 3D axes for display
x_min = y_min = z_min = -1
x_max = y_max = z_max = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
ax.view_init(30, 285)
plt.draw()
plt.pause(0.1)

while True:
	pf.weight()
	mle = pf.predict_mle()
	mean = pf.predict_mean()

	manifold_particles = cc.inverse_mapping(pf.particles)
	manifold_mle = cc.single_inverse_mapping(mle)
	manifold_mean = cc.single_inverse_mapping(mean)

	# Display
	ax.clear()
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)
	ax.set_zlim(z_min, z_max)
	ax.view_init(30, 285)
	ax.scatter(manifold_particles[:,0], manifold_particles[:,1], manifold_particles[:,2], cmap=plt.cm.cool, c=pf.weights, s=8**2)
	ax.scatter([manifold_mle[0]], [manifold_mle[1]], [manifold_mle[2]], color="black", marker="*", s=20**2)
	ax.scatter([manifold_mean[0]], [manifold_mean[1]], [manifold_mean[2]], color="black", marker="x", s=20**2)
	ax.scatter([ground_truth[0]], [ground_truth[1]], [ground_truth[2]], color="green", marker="+", s=20**2)
	plt.draw()
	plt.pause(0.1)

	pf.resample()
	pf.diffuse()