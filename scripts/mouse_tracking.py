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
# visualization.create_interactive_embedding_visulization(cc, 2)

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

def likelihood_one_zero(particle):
	xy, theta, deformation = unpack_particle(particle)
	manifold_deformation = cc.single_inverse_mapping(deformation)
	pose = compute_pose(xy, theta, manifold_deformation)
	pose = np.asarray(pose, dtype=int)
	for i in range(pose.shape[0]):
		if pose[i,0] < 0 or pose[i,1] < 0 or pose[i][0] >= camera_size[0] or pose[i][1] >= camera_size[1]:
			return 0
		elif mouse_dataset.test_images[test_ind][pose[i,1], pose[i,0]] < mouse_dataset.d2:
			return 1
	return 0

def likelihood_old(particle):
	xy, theta, deformation = unpack_particle(particle)
	manifold_deformation = cc.single_inverse_mapping(deformation)
	pose = compute_pose(xy, theta, manifold_deformation)
	pose = np.asarray(pose, dtype=int)
	total = 0
	for i in range(pose.shape[0]):
		if pose[i,0] < 0 or pose[i,1] < 0 or pose[i][0] >= camera_size[0] or pose[i][1] >= camera_size[1]:
			return 0
		elif mouse_dataset.test_images[test_ind][pose[i,1], pose[i,0]] < mouse_dataset.d2:
			total = total + 1
	return total

def diffuser(particle):
	xy, theta, deform = unpack_particle(particle)
	xy = xy + np.random.multivariate_normal(np.zeros(2), xy_var * np.eye(2))
	theta = (theta + np.random.normal(0, theta_var)) % (2 * np.pi)

	while True:
		delta = np.random.multivariate_normal(np.zeros(2), deformation_var * np.eye(2))
		if cc.check_domain([deform + delta])[0]:
			deform = deform + delta
			break

	return pack_particle(xy, theta, deform)

# pf = particle_filter.ParticleFilter(target_dim, n_particles, exploration_factor, keep_best, rand_sampler, trivial_likelihood, diffuser)
# pf = particle_filter.ParticleFilter(target_dim, n_particles, exploration_factor, keep_best, rand_sampler, likelihood_one_zero, diffuser)
pf = particle_filter.ParticleFilter(target_dim, n_particles, exploration_factor, keep_best, rand_sampler, likelihood_old, diffuser)

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

iter_num = 0
test_ind = test_start_ind
while True:
	pf.weight()
	mle = pf.predict_mle()
	mean = pf.predict_mean()

	unpacked_particles = [unpack_particle(p) for p in pf.particles]
	manifold_deformations = [cc.single_inverse_mapping(p[2]) for p in unpacked_particles]
	manifold_poses = [compute_pose(unpacked_particles[i][0], unpacked_particles[i][1], manifold_deformations[i]) for i in range(n_particles)]

	mean_xy, mean_theta, mean_deformation = unpack_particle(mean)
	mean_manifold_deformation = cc.single_inverse_mapping(mean_deformation)
	mean_pose = compute_pose(mean_xy, mean_theta, mean_manifold_deformation)

	# Display
	ax1.clear()
	ax2.clear()
	ax1.set_xlim(x_min, x_max)
	ax1.set_ylim(y_min, y_max)
	ax2.set_xlim(x_min, x_max)
	ax2.set_ylim(y_min, y_max)

	ax1.imshow(mouse_dataset.test_images[test_ind], cmap=plt.get_cmap('gray'), vmin=mouse_dataset.d1, vmax=mouse_dataset.d2)
	ax2.imshow(mouse_dataset.test_images[test_ind], cmap=plt.get_cmap('gray'), vmin=mouse_dataset.d1, vmax=mouse_dataset.d2)

	for i in range(n_particles):
		ax1.plot(manifold_poses[i][:,0], manifold_poses[i][:,1], c=plt.cm.cool(pf.weights[i]))
	ax2.plot(manifold_poses[pf.max_weight_ind][:,0], manifold_poses[pf.max_weight_ind][:,1], c="blue", label="MLE Particle")
	ax2.plot(mean_pose[:,0], mean_pose[:,1], c="green", label="Mean Particle")
	ax1.plot(mouse_dataset.test_poses[test_ind][:,0], mouse_dataset.test_poses[test_ind][:,1], c="red")
	ax2.plot(mouse_dataset.test_poses[test_ind][:,0], mouse_dataset.test_poses[test_ind][:,1], c="red", label="Ground Truth")
	ax2.legend()

	plt.draw()
	plt.pause(0.001)

	pf.resample()
	pf.diffuse()

	iter_num = iter_num + 1
	if track:
		test_ind = test_ind + 1