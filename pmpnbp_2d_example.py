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

# Dataset parameters
n_train = 200
restrict_deformations = True

# Manifold learning parameters
target_dim = 4
neighbors_k = 12

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
		circle = plt.Circle(self.position, radius=self.radius, color=color)
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
		points = self.get_vertices()
		ax.fill(points[:,0], points[:,1], color=color)

	def get_vertices(self):
		base_corner = self.position - (np.array([self.size[1] * np.cos(self.orientation + np.pi/2), self.size[1] * np.sin(self.orientation + np.pi/2)]) / 2)
		vec1 = np.array([self.size[0] * np.cos(self.orientation), self.size[0] * np.sin(self.orientation)])
		vec2 = np.array([self.size[1] * np.cos(self.orientation + np.pi/2), self.size[1] * np.sin(self.orientation + np.pi/2)])
		vertices = np.array([
			base_corner,
			base_corner + vec1,
			base_corner + vec1 + vec2,
			base_corner + vec2
		])
		return vertices

####################
# Create the Scene #
####################

# Make the noisy observations
scene_circles = []
scene_rectangles = []
for _ in range(n_circles):
	scene_circles.append(Circle())
for _ in range(n_rectangles):
	scene_rectangles.append(Rectangle())

# Construct ground truth
def make_ground_truth(angle_noises=None):
	if angle_noises is None:
		angle_noises = np.random.normal(loc=0, scale=gt_angle_var, size=2*len(gt_cardinal_direction_angles))
		if restrict_deformations:
			for i in range(len(gt_cardinal_direction_angles)):
				angle_noises[2*i] = angle_noises[2*i+1]
	gt_circle = Circle(position=dims/2, radius=circle_radius)
	gt_rectangles = []
	# Inner layer
	for i in range(len(gt_cardinal_direction_angles)):
		angle = gt_cardinal_direction_angles[i]
		orientation = angle + angle_noises[2*i]
		position = gt_circle.position + np.array([gt_inner_dist * np.cos(orientation), gt_inner_dist * np.sin(orientation)])
		gt_rectangles.append(Rectangle(position=position, orientation=orientation, size=rectangle_dims))

		orientation = angle + angle_noises[2*i] + angle_noises[2*i+1]
		position = position + np.array([gt_outer_dist * np.cos(orientation), gt_outer_dist * np.sin(orientation)])
		gt_rectangles.append(Rectangle(position=position, orientation=orientation, size=rectangle_dims))

	return gt_circle, gt_rectangles

gt_circle, gt_rectangles = make_ground_truth()
scene_circles.append(gt_circle)
for rectangle in gt_rectangles:
	scene_rectangles.append(rectangle)

def gt_to_state_vec(gt_circle, gt_rectangles):
	angles = np.array([r.orientation for r in gt_rectangles])
	state_vec = angles - np.vstack((gt_cardinal_direction_angles, gt_cardinal_direction_angles)).flatten("F")
	return state_vec

def state_vec_to_gt(state_vec):
	return make_ground_truth(angle_noises=state_vec)

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

for circle in scene_circles:
	circle.draw(ax, color="grey")
for rectangle in scene_rectangles:
	rectangle.draw(ax, color="grey")

gt_circle.draw(ax)
for rectangle in gt_rectangles:
	rectangle.draw(ax)

plt.draw()
# plt.pause(0.001)
plt.show()

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

#################
# Preprocessing #
#################

train = []
for _ in range(n_train):
	gt_circle, gt_rectangles = make_ground_truth()
	train.append(gt_to_state_vec(gt_circle, gt_rectangles))

# Compute the Isomap embedding
from sklearn.manifold import Isomap
embedding = Isomap(n_neighbors=neighbors_k, n_components=target_dim).fit_transform(train)

# Compute the Delaunay triangulation
from scipy.spatial import Delaunay
interpolator = Delaunay(embedding, qhull_options="QJ")

def compute_deformation(interpolator, deformation_coords):
	simplex_num = interpolator.find_simplex(deformation_coords)
	if simplex_num != -1:
		simplex_indices = interpolator.simplices[simplex_num]
		simplex = interpolator.points[simplex_indices]

		# Compute barycentric coordinates
		A = np.vstack((simplex.T, np.ones((1, target_dim+1))))
		b = np.vstack((deformation_coords.reshape(-1, 1), np.ones((1, 1))))
		try:
			b_coords = np.linalg.solve(A, b)
		except:
			print deformation_coords
			print simplex_num
			print simplex_indices
			print simplex
			print A
			print b
			exit(1)
		b = np.asarray(b_coords).flatten()

		# Interpolate the deformation
		mult_vec = np.zeros(len(train))
		mult_vec[simplex_indices] = b
		output = np.sum(np.matmul(np.diag(mult_vec), train), axis=0)
		return output
	else:
		print "Error: outside of convex hull!"
		raise ValueError

class Particle():
	def __init__(self, xy=None, deformation=None):
		if xy is None:
			self.xy = (np.random.randint(0, dims[0]),
			           np.random.randint(0, dims[1]))
		else:
			self.xy = xy

		if deformation is None:
			deformation_ind = np.random.randint(0, len(embedding))
			self.deformation = embedding[deformation_ind]
		else:
			self.deformation = deformation

		self.project_up()

		self.raw_weight = None
		self.normalized_weight = None

	def project_up(self):
		self.state_vec = compute_deformation(interpolator, self.deformation)