import numpy as np
import matplotlib.pyplot as plt

import coordinate_chart
import particle_filter
import visualization
import normalization
import utility

import data.mouse_dataset.mouse_dataset as mouse_dataset

#########################
# Experiment parameters #
#########################

# General parameters
track = False  # If true, track normally. If false, don't increase the frame number with each iteration.
               # False allows us to test only localizing in a single frame.

# Dataset parameters
n_train = 500        # Number of training samples to use
random_train = False # Optionally randomly select the training images from the whole dataset
test_start_ind = 0   # Can start the test sequence at a different index if desired
camera_size = 2 * np.array([mouse_dataset.cx, mouse_dataset.cy])

# Manifold learning
target_dim = 2   # The target dimension for ISOMAP.
neighbors_k = 12 # The number of neighbors used for ISOMAP.

# Particle filter
n_particles = 200         # Number of particles
exploration_factor = 0.25 # Fraction of particles used to explore
xy_var = 100              # Variance of diffusion noise added to particles' position component
theta_var = np.pi/32      # Variance of diffusion noise added to particles' orientation component
deformation_var = 250     # Variance of diffusion noise added to particles' deformation component
keep_best = True          # Keep the best guess unchanged

###########
# Dataset #
###########

if random_train:
	mouse_dataset.load_train()
	train_inds = np.random.choice(mouse_dataset.n_train, n_train, replace=False)
else:
	mouse_dataset.load_train(n_train)
	train_inds = np.arange(n_train)
mouse_dataset.load_test()

train_data = np.array([mouse_dataset.train_poses[i] for i in train_inds])[:,:,:2]
normalized_train_data = normalization.normalize_pointcloud_2d(train_data)

####################
# Coordinate Chart #
####################

cc = coordinate_chart.CoordinateChart(normalized_train_data.reshape(n_train,-1), target_dim, neighbors_k)
visualization.create_interactive_embedding_visulization(cc, 2)

#########################
# Particle Filter Setup #
#########################

def pack_particle(xy,theta,deform):
	return np.concatenate((xy,[theta],deform))

def unpack_particle(particle):
	return particle[:2], particle[2], particle[3:]

def compute_pose(xy,theta,transformed_point):
	point_cloud = transformed_point.reshape(-1,2)
	c,s = np.cos(theta), np.sin(theta)
	rot_mat = np.array([[c, -s], [s, c]])
	point_cloud = np.matmul(point_cloud, rot_mat.T) + xy
	return point_cloud

def rand_sampler(n):
	xy = np.zeros((n,2))
	theta = np.zeros(n)
	deform = np.zeros((n,cc.target_dim))
	points = np.zeros((n,2+1+cc.target_dim))

	xy[:,0] = np.random.uniform(low=0, high=camera_size[0], size=n)
	xy[:,1] = np.random.uniform(low=0, high=camera_size[1], size=n)
	theta = np.random.uniform(low=0, high=2*np.pi, size=n)
	deform = cc.uniform_sample(n)

	points[:,0:2] = xy
	points[:,2] = theta
	points[:,3:] = deform
	return points

def trivial_likelihood(particle):
	return 1

def diffuser(particle):
	xy, theta, deform = unpack_particle(particle)
	xy = xy + np.random.multivariate_normal(np.zeros(2), xy_var * np.eye(2))
	theta = (theta + np.random.normal(0, theta_var)) % (2 * np.pi)

	while True:
		delta = np.random.multivariate_normal(np.zeros(2), deformation_var * np.eye(2))
		if cc.tri.check_domain([deform + delta])[0]:
			deform = deform + delta
			break

	return pack_particle(xy, theta, deform)

pf = particle_filter.ParticleFilter(target_dim, n_particles, exploration_factor, keep_best, rand_sampler, trivial_likelihood, diffuser)

########################
# Particle Filter Loop #
########################

# Create axes for display
x_min = y_min = 0
x_max = camera_size[0]
y_max = camera_size[1]

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# Set the perspective
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
visualization.maximize_window()
plt.draw()
plt.pause(0.001)