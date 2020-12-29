import numpy as np

# Functions to normalize point clouds, so that they're centered, and all have the same orientation

def normalize_pointcloud_2d(data, centering_idx=0, orientation_idx=1):
	# The data should be of shape (n_clouds, n_points, 2)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud, and must be greater than 1
	# centering_idx is the index of the point which is centered at 0
	# orientation_idx is the index of the point used for uniform rotating
	# These indices must be distinct
	# Returns (normalized_data, inverse_transformation_matrix)
	pass

def normalize_pointcloud_3d(data, centering_idx=0, orientation_idx_1=1, orientation_idx_2=2):
	# The data should be of shape (n_clouds, n_points, 3)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud, and must be greater than 2
	# centering_idx is the index of the point which is centered at 0
	# orientation_idx_1, orientation_idx_2 are the indices of the points used for uniform rotating
	# These indices must be distinct, and should be chosen so that they aren't collinear
	# Returns (normalized_data, inverse_transformation_matrix)
	pass

def center_pointcloud(data, centering_idx=0):
	# The data should be of shape (n_clouds, n_points, n_dimensions)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud
	# n_dimensions is arbitrary
	# centering_idx is the index of the point which is centered at 0
	# Returns (centered_data, inverse_transformation_matrix)
	pass