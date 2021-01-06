import numpy as np

# Functions to normalize point clouds, so that they're centered, and all have the same orientation

def normalize_pointcloud_2d(data, centering_idx=0, orientation_idx=1):
	# The data should be of shape (n_clouds, n_points, 2)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud, and must be greater than 1
	# centering_idx is the index of the point which is centered at 0
	# orientation_idx is the index of the point used for uniform rotating
	# These indices must be distinct

	# First, center the data
	centered_data, translation = center_pointcloud(data, centering_idx)

	# Next, compute the angle for each point cloud (from the x-axis to orientation_idx)
	angs = np.arctan2(centered_data[:,orientation_idx,1], centered_data[:,orientation_idx,0])

	# Next, create rotation matrices for each cloud
	cos = np.cos(-angs)
	sin = np.sin(-angs)
	mats = np.swapaxes(np.array([[cos, -sin], [sin, cos]]), 0, 2)

	# Finally, apply the rotation matrices to each cloud
	# TODO: find a way to vectorize this
	normalized_data = np.zeros(centered_data.shape)
	for i in range(centered_data.shape[0]):
		normalized_data[i,:] = np.matmul(centered_data[i,:], mats[i])

	return normalized_data, translation, mats

def normalize_pointcloud_3d(data, centering_idx=0, orientation_idx_1=1, orientation_idx_2=2):
	# The data should be of shape (n_clouds, n_points, 3)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud, and must be greater than 2
	# centering_idx is the index of the point which is centered at 0
	# orientation_idx_1, orientation_idx_2 are the indices of the points used for uniform rotating
	# These indices must be distinct, and should be chosen so that they aren't collinear

	# Reference: https://math.stackexchange.com/questions/856666/how-can-i-transform-a-3d-triangle-to-xy-plane

	# First, center the data
	centered_data, translation = center_pointcloud(data, centering_idx)

	# Next, create rotation matrices for each cloud
	us = centered_data[:,orientation_idx_1]
	Us = us / np.linalg.norm(us, axis=1).reshape(-1,1)
	ws = np.cross(us, centered_data[:,orientation_idx_2])
	Ws = ws / np.linalg.norm(ws, axis=1).reshape(-1,1)
	Vs = np.cross(Us,Ws)
	mats = np.swapaxes(np.array([Us, Vs, Ws]).T, 0, 1)

	# Finally, apply the rotation matrices to each cloud
	# TODO: find a way to vectorize this
	normalized_data = np.zeros(centered_data.shape)
	for i in range(centered_data.shape[0]):
		normalized_data[i,:] = np.matmul(centered_data[i,:], mats[i])

	return normalized_data, translation, mats

def center_pointcloud(data, centering_idx=0):
	# The data should be of shape (n_clouds, n_points, n_dimensions)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud
	# n_dimensions is arbitrary
	# centering_idx is the index of the point which is centered at 0
	offset = data[:,[centering_idx],:] # Shape (n_clouds, 1, n_dimension)
	offset_cloud = np.tile(offset, (1, data.shape[1], 1)) # Shape (n_clouds, n_points, n_dimensions)
	return data - offset_cloud, offset