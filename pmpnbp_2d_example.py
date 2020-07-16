import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry.point import Point
import shapely.affinity
from shapely.geometry import Polygon

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

	def get_vertices(self):
		base_corner = self.position - np.array([0, rectangle_dims[1]/2])
		vec1 = np.array([self.size[0] * np.cos(self.orientation), self.size[1] * np.sin(self.orientation)])
		vec2 = np.array([self.size[0] * np.cos(self.orientation + np.pi/2), self.size[1] * np.sin(self.orientation + np.pi/2)])
		vertices = [
			base_corner,
			base_corner + vec1,
			base_corner + vec1 + vec2,
			base_corner + vec2
		]
		return vertices

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
def make_ground_truth():
	gt_circle = Circle(position=dims/2, radius=circle_radius)
	gt_rectangles = []
	# Inner layer
	for angle in gt_cardinal_direction_angles:
		angle_noise = np.random.normal(loc=0, scale=gt_angle_var)

		orientation = angle + angle_noise
		position = gt_circle.position + np.array([gt_inner_dist * np.cos(orientation), gt_inner_dist * np.sin(orientation)])
		gt_rectangles.append(Rectangle(position=position, orientation=orientation, size=rectangle_dims))

		orientation = angle + 2 * angle_noise
		position = position + np.array([gt_outer_dist * np.cos(orientation), gt_outer_dist * np.sin(orientation)])
		gt_rectangles.append(Rectangle(position=position, orientation=orientation, size=rectangle_dims))

	return gt_circle, gt_rectangles

gt_circle, gt_rectangles = make_ground_truth()

#####################
# Display the Scene #
#####################

fig, ax = plt.subplots(1, 1)
ax.set_xlim((0, dims[0]))
ax.set_ylim((0, dims[1]))
ax.set_aspect('equal')
ax.set_facecolor("black")
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

for circle in circles:
	circle.draw(ax, color="grey")
for rectangle in rectangles:
	rectangle.draw(ax, color="grey")

gt_circle.draw(ax)
for rectangle in gt_rectangles:
	rectangle.draw(ax)

plt.draw()
plt.pause(0.001)

#######################
# Shape IOU Functions #
#######################

def intersection_over_union(shape1, shape2):
	if isinstance(shape1, Circle) and isinstance(shape2, Circle):
		return iou_circle_circle(shape1, shape2)
	elif isinstance(shape1, Circle) and isinstance(shape2, Rectangle):
		return iou_circle_rectangle(shape1, shape2)
	elif isinstance(shape1, Rectangle) and isinstance(shape2, Circle):
		return iou_circle_rectangle(shape2, shape1)
	elif isinstance(shape1, Rectangle) and isinstance(shape2, Rectangle):
		return iou_rectangle_rectangle(shape1, shape2)

def circle_area(circle):
	return np.pi * (circle.radius**2)

def rectangle_area(rectangle):
	return rectangle.size[0] * rectangle.size[1]

def iou_circle_circle(circle1, circle2):
	# https://mathworld.wolfram.com/Circle-CircleIntersection.html
	R = circle1.radius
	r = circle2.radius
	d = np.linalg.norm(circle2.position - circle1.position)
	
	x = ((d**2) - (r**2) + (R**2)) / (2 * d)
	
	d1 = x
	d2 = d - x
	A = ((r**2) * np.arccos(((d**2) + (r**2) - (R**2)) / (2 * d * r)))\
	  + ((R**2) * np.arccos(((d**2) + (R**2) - (r**2)) / (2 * d * R)))\
	  - (0.5 * np.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)))

	intersection = A
	union = circle_area(circle1) + circle_area(circle2) - intersection
	return intersection / union

def iou_circle_rectangle(circle, rectangle):
	shapely_rectangle = Polygon(rectangle.get_vertices())
	shapely_circle = Point(circle.position).buffer(circle.radius)

	intersection = shapely_rectangle.intersection(shapely_circle).area
	union = circle_area(circle) + rectangle_area(rectangle) - intersection
	return intersection / union

def iou_rectangle_rectangle(rectangle1, rectangle2):
	shapely_rectangle1 = Polygon(rectangle1.get_vertices())
	shapely_rectangle2 = Polygon(rectangle2.get_vertices())

	intersection = shapely_rectangle1.intersection(shapely_rectangle2).area
	union = rectangle_area(rectangle1) + rectangle_area(rectangle2) - intersection
	return intersection / union