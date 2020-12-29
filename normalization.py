import numpy as np

# Functions to normalize point clouds, so that they're centered, and all have the same orientation

def normalize_pointcloud_2d(data, centering_idx=0, orientation_idx=1):
	# The data should be of shape (n_clouds, n_points, 2)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud, and must be greater than 1
	# centering_idx is the index of the point which is centered at 0
	# orientation_idx is the index of the point used for uniform rotating
	# These indices must be distinct
	pass

def normalize_pointcloud_3d(data, centering_idx=0, orientation_idx_1=1, orientation_idx_2=2):
	# The data should be of shape (n_clouds, n_points, 3)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud, and must be greater than 2
	# centering_idx is the index of the point which is centered at 0
	# orientation_idx_1, orientation_idx_2 are the indices of the points used for uniform rotating
	# These indices must be distinct, and should be chosen so that they aren't collinear
	pass

def center_pointcloud(data, centering_idx=0):
	# The data should be of shape (n_clouds, n_points, n_dimensions)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud
	# n_dimensions is arbitrary
	# centering_idx is the index of the point which is centered at 0
	offset = data[:,[centering_idx],:] # Shape (n_clouds, 1, n_dimension)
	offset_cloud = np.tile(offset, (1, data.shape[1], 1)) # Shape (n_clouds, n_points, n_dimensions)
	return data - offset_cloud