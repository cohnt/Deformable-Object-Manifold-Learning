import numpy as np
import matplotlib
import matplotlib.pyplot as plt

##############
# Parameters #
##############

# Scene setup
dims = np.array([20, 20])
circle_radius = 0.5
circle_noise_var = 0.05
n_circles = 25
rectangle_dims = np.array([1.25, 0.25])
rectangle_noise_cov = np.array([[0.05, 0], [0, 0.025]])
n_rectangles = 100

# Ground truth rules
gt_inner_dist = circle_radius + 1.0
gt_outer_dist = rectangle_dims[0] + 1.0
gt_cardinal_direction_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
gt_angle_var = np.pi / 16

#########
# Types #
#########

def rad2deg(r):
	return r * 180 / np.pi

class Circle():
	def __init__(self, position=None, radius=None):
		if position is None:
			self.position = np.random.uniform([0, 0], dims)
		else:
			self.position = position

		if radius is None:
			self.radius = circle_radius + np.random.normal(loc=0, scale=circle_noise_var)
		else:
			self.radius = radius

	def draw(self, ax, color="white"):
		circle = plt.Circle(self.position, radius=self.radius, facecolor=color)
		ax.add_patch(circle)

class Rectangle():
	def __init__(self, position=None, orientation=None, size=None):
		if position is None:
			self.position = np.random.uniform([0, 0], dims)
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

	def draw(self, ax, color="white"):
		render_offset = np.array([0, rectangle_dims[1]/2])
		rectangle = plt.Rectangle(self.position-render_offset, self.size[0], self.size[1], angle=rad2deg(self.orientation), facecolor=color)
		ax.add_patch(rectangle)

####################
# Create the Scene #
####################

# Make the noisy observations
circles = []
rectangles = []
for _ in range(n_circles):
	circles.append(Circle())
for _ in range(n_rectangles):
	rectangles.append(Rectangle())

# Construct ground truth
gt_circle = Circle(position=dims/2, radius=circle_radius)
gt_rectangles = []
# Inner layer
for angle in gt_cardinal_direction_angles:
	orientation = angle + np.random.normal(loc=0, scale=gt_angle_var)
	position = gt_circle.position + np.array([gt_inner_dist * np.cos(orientation), gt_inner_dist * np.sin(orientation)])
	gt_rectangles.append(Rectangle(position=position, orientation=orientation, size=rectangle_dims))

	orientation = orientation + np.random.normal(loc=0, scale=gt_angle_var)
	position = position + np.array([gt_outer_dist * np.cos(orientation), gt_outer_dist * np.sin(orientation)])
	gt_rectangles.append(Rectangle(position=position, orientation=orientation, size=rectangle_dims))

#####################
# Display the Scene #
#####################

fig, ax = plt.subplots(1, 1)
ax.set_xlim((0, dims[0]))
ax.set_ylim((0, dims[1]))
ax.set_aspect('equal')
ax.set_facecolor("black")

for circle in circles:
	circle.draw(ax, color="grey")
for rectangle in rectangles:
	rectangle.draw(ax, color="grey")

gt_circle.draw(ax)
for rectangle in gt_rectangles:
	rectangle.draw(ax)

plt.show()