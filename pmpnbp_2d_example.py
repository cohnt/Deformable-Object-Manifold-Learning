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
gt_angle_var = np.pi / 24

# Dataset parameters
n_train = 200
restrict_deformations = True

# Manifold learning parameters
target_dim = 4
neighbors_k = 12

# Particle filter parameters
n_particles = 100
exploration_factor = 0.0
position_var = 0.25
deformation_var = 0.1

matplotlib.rcParams.update({'font.size': 22})

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

	def draw(self, ax, color="white", alpha=1.0):
		circle = plt.Circle(self.position, radius=self.radius, color=color, alpha=alpha)
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

	def draw(self, ax, color="white", alpha=1.0):
		points = self.get_vertices()
		ax.fill(points[:,0], points[:,1], color=color, alpha=alpha)

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
def make_thingy(angle_noises=None, position=None):
	if angle_noises is None:
		angle_noises = np.random.normal(loc=0, scale=gt_angle_var, size=2*len(gt_cardinal_direction_angles))
		if restrict_deformations:
			for i in range(len(gt_cardinal_direction_angles)):
				angle_noises[2*i] = angle_noises[2*i+1]
	if position is None:
		position = dims/2

	circle = Circle(position=position, radius=circle_radius)
	rectangles = []
	for i in range(len(gt_cardinal_direction_angles)):
		angle = gt_cardinal_direction_angles[i]
		orientation = angle + angle_noises[2*i]
		position = circle.position + np.array([gt_inner_dist * np.cos(orientation), gt_inner_dist * np.sin(orientation)])
		rectangles.append(Rectangle(position=position, orientation=orientation, size=rectangle_dims))

		orientation = angle + angle_noises[2*i] + angle_noises[2*i+1]
		position = position + np.array([gt_outer_dist * np.cos(orientation), gt_outer_dist * np.sin(orientation)])
		rectangles.append(Rectangle(position=position, orientation=orientation, size=rectangle_dims))

	return circle, rectangles

gt_circle, gt_rectangles = make_thingy()
scene_circles.append(gt_circle)
for rectangle in gt_rectangles:
	scene_rectangles.append(rectangle)

def thingy_to_state_vec(circle, rectangles):
	angles = np.array([r.orientation for r in rectangles])
	state_vec = angles - np.vstack((gt_cardinal_direction_angles, gt_cardinal_direction_angles)).flatten("F")
	return state_vec

def state_vec_to_thingy(state_vec):
	return make_thingy(angle_noises=state_vec)

#####################
# Display the Scene #
#####################

def draw_scene(ax):
	for circle in scene_circles:
		circle.draw(ax, color="grey")
	for rectangle in scene_rectangles:
		rectangle.draw(ax, color="grey")

	gt_circle.draw(ax)
	for rectangle in gt_rectangles:
		rectangle.draw(ax)

def clear(ax):
	ax.cla()
	ax.set_xlim((0, dims[0]))
	ax.set_ylim((0, dims[1]))
	ax.set_aspect('equal')
	ax.set_facecolor("black")

fig, ax = plt.subplots(1, 1)
clear(ax)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

draw_scene(ax)
plt.draw()
plt.pause(0.001)
# plt.show()

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

def iou_circle_circle(circle1, circle2):
	shapely_circle1 = Point(circle1.position).buffer(circle1.radius)
	shapely_circle2 = Point(circle2.position).buffer(circle2.radius)

	intersection = shapely_circle1.intersection(shapely_circle2).area
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
	circle, rectangles = make_thingy()
	train.append(thingy_to_state_vec(circle, rectangles))

# Compute the Isomap embedding
from sklearn.manifold import Isomap
embedding = Isomap(n_neighbors=neighbors_k, n_components=target_dim).fit_transform(train)

print np.min(embedding)
print np.max(embedding)

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

#########################
# Particle Filter Setup #
#########################

class Particle():
	def __init__(self, position=None, deformation=None):
		if position is None:
			self.position = np.random.uniform([0, 0], dims)
		else:
			self.position = position

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
		self.circle, self.rectangles = make_thingy(angle_noises=self.state_vec, position=self.position)

	def draw(self, ax, alpha=0.25):
		color = plt.cm.Spectral(1.0 - self.raw_weight)
		self.circle.draw(ax, color, alpha)
		for rectangle in self.rectangles:
			rectangle.draw(ax, color, alpha)

def shape_weight(shape):
	ious = []
	for circle in scene_circles:
		ious.append(intersection_over_union(shape, circle))
	for rectangle in scene_rectangles:
		ious.append(intersection_over_union(shape, rectangle))
	return np.max(ious)

def particle_weight(particle):
	weights = []
	weights.append(shape_weight(particle.circle))
	for rectangle in particle.rectangles:
		weights.append(shape_weight(rectangle))
	return np.mean(weights)

particles = [Particle() for _ in range(n_particles)]
iter_num = 0

try:
	while True:
		iter_num = iter_num + 1
		print "Iteration %d" % iter_num

		if iter_num > 100:
			break

		# Weight particles
		weights = []
		for p in particles:
			p.raw_weight = particle_weight(p)
			weights.append(p.raw_weight)
		weights = np.asarray(weights)
		normalization_factor = 1.0 / np.sum(weights)
		normalized_weights = []
		for p in particles:
			w = p.raw_weight * normalization_factor
			p.normalized_weight = w
			normalized_weights.append(w)
		max_normalized_weight = np.max(normalized_weights)
		max_normalized_weight_ind = np.argmax(normalized_weights)

		# Display
		clear(ax)
		draw_scene(ax)
		for p in particles:
			p.draw(ax)

		ax.scatter([particles[max_normalized_weight_ind].circle.position[0]], [particles[max_normalized_weight_ind].circle.position[1]], color="red", zorder=2)
		for rectangle in particles[max_normalized_weight_ind].rectangles:
			ax.scatter(rectangle.get_vertices()[:,0], rectangle.get_vertices()[:,1], color="red", zorder=2)
		ax.scatter([gt_circle.position[0]], [gt_circle.position[1]], color="green", zorder=2)
		for rectangle in gt_rectangles:
			ax.scatter(rectangle.get_vertices()[:,0], rectangle.get_vertices()[:,1], color="green", zorder=2)

		plt.draw()
		plt.pause(0.001)
		plt.savefig("iteration%03d.png" % iter_num)

		# Resample
		newParticles = []
		cs = np.cumsum(normalized_weights)
		step = 1/float((n_particles * (1-exploration_factor))+1)
		chkVal = step
		chkIdx = 0
		newParticles.append(particles[max_normalized_weight_ind])
		for i in range(1, int(np.ceil(n_particles * (1-exploration_factor)))):
			while cs[chkIdx] < chkVal:
				chkIdx = chkIdx + 1
			chkVal = chkVal + step
			newParticles.append(Particle(position=particles[chkIdx].position,
			                             deformation=particles[chkIdx].deformation))
		for i in range(len(newParticles), n_particles):
			newParticles.append(Particle())

		# Add noise
		particles = newParticles
		for p in particles:
			p.position = p.position + np.random.multivariate_normal(np.zeros(2), position_var*np.eye(2))

			while True:
				delta = np.random.multivariate_normal(np.zeros(target_dim), deformation_var*np.eye(target_dim))
				if interpolator.find_simplex(p.deformation + delta) != -1:
					p.deformation = p.deformation + delta
					break

			p.project_up()
except KeyboardInterrupt:
	pass

import os
os.system('ffmpeg -f image2 -r 1/0.5 -i iteration\%03d.png -c:v libx264 -pix_fmt yuv420p out.mp4')