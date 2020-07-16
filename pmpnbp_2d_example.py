import numpy as np
import matplotlib
import matplotlib.pyplot as plt

##############
# Parameters #
##############

# Scene setup
dims = np.array([10, 10])
circle_radius = 0.5
circle_noise_var = 0.1
n_circles = 8
rectangle_dims = np.array([1.25, 0.25])
rectangle_noise_cov = np.array([[0.1, 0], [0, 0.05]])
n_rectangles = 20

#########
# Types #
#########

class Circle():
	def __init__(self, position=None, radius=None):
		if position is None:
			self.position = np.array([
				np.random.uniform([0, 0], dims)
			])
		else:
			self.position = position

		if radius is None:
			self.radius = circle_radius + np.random.normal(loc=0, scale=circle_noise_var)
		else:
			self.radius = radius

class Rectangle():
	def __init__(self, position=None, orientation=None, size=None):
		if position is None:
			self.position = np.array([
				np.random.uniform([0, 0], dims)
			])
		else:
			self.position = position

		if orientation is None:
			self.orientation = np.random.uniform(0, 2*np.pi)
		else:
			self.orientation = orientation

		if size is None:
			self.size = rectangle_dims + np.random.multivariate_normal(mean=[0, 0], cov=rectangle_noise_cov)
		else:
			self.size = size

####################
# Create the Scene #
####################

