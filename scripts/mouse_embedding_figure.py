import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import coordinate_chart
import normalization

import data.mouse_dataset.mouse_dataset as mouse_dataset

# Dataset parameters
n_train = 500        # Number of training samples to use
random_train = False # Optionally randomly select the training images from the whole dataset
camera_size = 2 * np.array([mouse_dataset.cx, mouse_dataset.cy])

# Manifold learning
target_dim = 2   # The target dimension for ISOMAP.
neighbors_k = 12 # The number of neighbors used for ISOMAP.

# Visualization
embedding_point_radius = 20
vis_point_radius = 20
vis_pc_radius = 20

###########
# Dataset #
###########

if random_train:
	mouse_dataset.load_train()
	train_inds = np.random.choice(mouse_dataset.n_train, n_train, replace=False)
else:
	mouse_dataset.load_train(n_train)
	train_inds = np.arange(n_train)

train_data = np.array([mouse_dataset.train_poses[i] for i in train_inds])[:,:,:2]
normalized_train_data, translations, rotations = normalization.normalize_pointcloud_2d(train_data)

train_clouds = [mouse_dataset.train_clouds[i][:,[1,0]] for i in train_inds]
normalized_train_clouds = [np.matmul(train_clouds[i] - translations[i], rotations[i]) for i in range(n_train)]

####################
# Coordinate Chart #
####################

cc = coordinate_chart.CoordinateChart(normalized_train_data.reshape(n_train,-1), target_dim, neighbors_k)

########################
# Visualization/Figure #
########################

def create_interactive_embedding_visulization(cc, point_cloud_dim):
	fig = plt.figure()
	ax0 = fig.add_subplot(1, 2, 1)
	ax1 = fig.add_subplot(1, 2, 2)
	axes = [ax0, ax1]

	points = axes[0].scatter(cc.embedding[:,0], cc.embedding[:,1], c="grey", s=20**2)
	xlim = axes[0].get_xlim()
	ylim = axes[0].get_ylim()

	every_possible_point = np.concatenate(normalized_train_clouds)
	mfd_xlims = (np.min(every_possible_point[:,0]), np.max(every_possible_point[:,0]))
	mfd_ylims = (np.min(every_possible_point[:,1]), np.max(every_possible_point[:,1]))

	def hover(event):
		xy = np.array([event.xdata, event.ydata])
		if xy[0] is None or xy[1] is None:
			return

		# Find the nearest embedding point
		nearest_idx = -1
		nearest_dist = np.inf
		for i in range(len(cc.embedding)):
			if np.linalg.norm(cc.embedding[i] - xy) < nearest_dist:
				nearest_dist = np.linalg.norm(cc.embedding[i] - xy)
				nearest_idx = i

		# Display the embedding and highlighted point
		axes[0].clear()
		axes[0].scatter(cc.embedding[:,0], cc.embedding[:,1], c="grey", s=embedding_point_radius**2) # Draw the embedding points
		axes[0].scatter([cc.embedding[nearest_idx,0]], [cc.embedding[nearest_idx,1]], c="blue", s=embedding_point_radius**2) # Draw the nearest point
		axes[0].set_xlim(xlim) # Fix axes limits
		axes[0].set_ylim(ylim)

		# Transform the selected point
		point_cloud = normalized_train_data[nearest_idx]
		mouse_cloud = normalized_train_clouds[nearest_idx]

		# Draw the original point cloud
		axes[1].clear()
		axes[1].scatter(mouse_cloud[:,0], mouse_cloud[:,1], s=vis_pc_radius**2, c="grey")
		axes[1].scatter(point_cloud[:,0], point_cloud[:,1], s=vis_point_radius**2, c="blue")
		axes[1].plot(point_cloud[:,0], point_cloud[:,1], c="blue")
		axes[1].set_xlim(mfd_xlims)
		axes[1].set_ylim(mfd_ylims)

		# Update the figure
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect('motion_notify_event', hover)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.show()

create_interactive_embedding_visulization(cc, 2)