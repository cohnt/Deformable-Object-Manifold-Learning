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
		rectangle = plt.Rectangle(self.position, self.size[0], self.size[1], angle=rad2deg(self.orientation), facecolor=color)
		ax.add_patch(rectangle)

####################
# Create the Scene #
####################

circles = []
rectangles = []
for _ in range(n_circles):
	circles.append(Circle())
for _ in range(n_rectangles):
	rectangles.append(Rectangle())

fig, ax = plt.subplots(1, 1)
ax.set_xlim((0, dims[0]))
ax.set_ylim((0, dims[1]))
ax.set_facecolor("black")
for circle in circles:
	circle.draw(ax)
for rectangle in rectangles:
	rectangle.draw(ax)
plt.show()