import numpy as np

# Functions to normalize point clouds, so that they're centered, and all have the same orientation

def normalize_pointcloud_2d(data):
	# The data should be of shape (n_clouds, n_points, 2)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud
	# Returns (normalized_data, inverse_transformation_matrix)
	pass

def normalize_pointcloud_3d(data):
	# The data should be of shape (n_clouds, n_points, 3)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud
	# Returns (normalized_data, inverse_transformation_matrix)
	pass

def center_pointcloud(data):
	# The data should be of shape (n_clouds, n_points, n_dimensions)
	# n_clouds is the number of point cloud samples
	# n_points is the number of points in a point cloud
	# n_dimensions is arbitrary
	# Returns (centered_data, inverse_transformation_matrix)
	pass