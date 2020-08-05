import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from shapely.geometry import Polygon, Point

img_size = (50, 50)
mask = np.zeros(img_size)
blob = np.array([
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
	[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
	[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
])
blob_size = blob.shape
blob_min_x = 10
blob_max_x = blob_min_x + blob.shape[0] + 1
blob_min_y = 10
blob_max_y = blob_min_y + blob.shape[1] + 1
mask[blob_min_x:blob_min_x+blob.shape[0], blob_min_y:blob_min_y+blob_size[1]] = blob
blob_points = np.flip(np.transpose(blob.nonzero()), axis=1) +  [blob_min_x, blob_min_y]

spline_n_points = 40
spline_init_radius = 10
centroid = np.flip(np.array(center_of_mass(mask)))
angles = np.linspace(0, 2*np.pi, spline_n_points+1)[0:-1]
control_points = (spline_init_radius * np.vstack((np.cos(angles), np.sin(angles))).T) + centroid
render_points_per_segment = 2

compute_both_ways = True
spline_mode = "centripetal" # uniform, centripetal, or chordal
spline_alpha = 0 if spline_mode == "uniform" else (0.5 if spline_mode == "centripetal" else 1)
regularization_mode = "distance" # none, variance, distance, or curvature
reduce_learning_rate = 1.0 # 1.0 for no decrease
move_point = True
deterministic_move_point = move_point and False

class CatmullRomSplineSegment():
	def __init__(self, P0, P1, P2, P3, alpha=spline_alpha):
		self.P0 = np.asarray(P0)
		self.P1 = np.asarray(P1)
		self.P2 = np.asarray(P2)
		self.P3 = np.asarray(P3)
		self.alpha = alpha

		def tj(ti, Pi, Pj):
			xi, yi = Pi
			xj, yj = Pj
			return ((xj-xi)**2 + (yj-yi)**2)**alpha + ti

		self.t0 = 0
		self.t1 = tj(self.t0, self.P0, self.P1)
		self.t2 = tj(self.t1, self.P1, self.P2)
		self.t3 = tj(self.t2, self.P2, self.P3)
		self.t_length = self.t2 - self.t1

	def __call__(self, T):
		T = np.asarray(T).reshape(-1, 1)
		t = ((self.t2 - self.t1) * T) + self.t1
		A1 = (self.t1-t)/(self.t1-self.t0)*self.P0 + (t-self.t0)/(self.t1-self.t0)*self.P1
		A2 = (self.t2-t)/(self.t2-self.t1)*self.P1 + (t-self.t1)/(self.t2-self.t1)*self.P2
		A3 = (self.t3-t)/(self.t3-self.t2)*self.P2 + (t-self.t2)/(self.t3-self.t2)*self.P3
		B1 = (self.t2-t)/(self.t2-self.t0)*A1 + (t-self.t0)/(self.t2-self.t0)*A2
		B2 = (self.t3-t)/(self.t3-self.t1)*A2 + (t-self.t1)/(self.t3-self.t1)*A3
		C = (self.t2-t)/(self.t2-self.t1)*B1 + (t-self.t1)/(self.t2-self.t1)*B2
		return C.reshape(-1, 2)

class CatmullRomSpline():
	def __init__(self, points):
		self.points = points
		self.segments = []
		self.segments.append(CatmullRomSplineSegment(points[-1], points[0], points[1], points[2]))
		for i in range(len(points) - 3):
			self.segments.append(CatmullRomSplineSegment(points[i], points[i+1], points[i+2], points[i+3]))
		self.segments.append(CatmullRomSplineSegment(points[-3], points[-2], points[-1], points[0]))
		self.segments.append(CatmullRomSplineSegment(points[-2], points[-1], points[0], points[1]))
		self.segments = np.array(self.segments)

		self.t_lengths = np.array([segment.t_length for segment in self.segments])
		self.t_benchmarks = np.cumsum(self.t_lengths)
		self.total_length = self.t_benchmarks[-1]
		self.t_benchmarks = np.append(self.t_benchmarks, [0])
		# Having a 0 at the end makes sure that accessing the -1 element returns a 0 offset.

		self.interior_points = self.rasterize()

	def __call__(self, T):
		T = np.asarray(T).reshape(-1, 1)
		overall_t = T * self.total_length
		segment_idx = np.argmin(self.t_benchmarks < overall_t, axis=1)
		local_t = (overall_t - self.t_benchmarks[segment_idx-1].reshape(-1, 1)) / self.t_lengths[segment_idx].reshape(-1, 1)
		output = np.array([segment(t) for segment, t, in zip(self.segments[segment_idx], local_t)])
		return output.reshape(-1, 2)

	def rasterize(self, t_resolution=render_points_per_segment):
		points = []
		for segment in self.segments:
			Tvals = np.linspace(0, 1, t_resolution, endpoint=False)
			for t in Tvals:
				points.append(segment(t))
		points = np.array(points).reshape(-1, 2)
		polygon = Polygon(points)
		self.min_x = int(np.floor(np.min(points[:,0])))
		self.max_x = int(np.ceil(np.max(points[:,0])))
		self.min_y = int(np.floor(np.min(points[:,1])))
		self.max_y = int(np.ceil(np.max(points[:,1])))
		points = []
		for x in range(self.min_x, self.max_x+1):
			for y in range(self.min_y, self.max_y+1):
				point = Point(x, y)
				if polygon.contains(point):
					points.append([x, y])
		return np.array(points)

cms = CatmullRomSpline(control_points)

Tvals = np.linspace(0, 1, 100).reshape(-1, 1)
points = cms(Tvals)
plt.imshow(mask)
plt.plot(points[:,0], points[:,1])
plt.scatter(control_points[:,0], control_points[:,1])
points = cms.interior_points
plt.scatter(points[:,0], points[:,1])

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.draw()
plt.pause(0.001)
plt.savefig("iteration%03d_pre.png" % 0)
plt.savefig("iteration%03d_post.png" % 0)

def iou(spline):
	intersection = 0.
	for p1 in blob_points:
		for p2 in spline.interior_points:
			if (p1 == p2).all():
				intersection = intersection + 1
	union = len(blob_points) + len(spline.interior_points) - intersection
	return intersection / union

def loss(spline):
	if regularization_mode == "variance":
		return iou(spline) - 0.01 * distances_variance(spline)
	elif regularization_mode == "distance":
		return iou(spline) + 0.01 * distance_regularization_penalty(spline)
	elif regularization_mode == "curvature":
		return iou(spline) + curvature_penalty(spline)
	else:
		return iou(spline)

def distances_variance(spline):
	dists = []
	for i in range(len(spline.points) - 1):
		dists.append(np.linalg.norm(spline.points[i] - spline.points[i+1]))
	dists.append(np.linalg.norm(spline.points[-1] - spline.points[0]))
	return np.var(dists)

def curvature_penalty(spline):
	smooths = np.zeros(len(spline.points))
	for i in range(1, len(spline.points)-1):
		v1 = spline.points[i] - spline.points[i-1]
		v2 = spline.points[i+1] - spline.points[i]
		angle = np.arccos(np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))))
		smoothness = np.abs(angle - (np.pi / 2)) / (np.pi / 2)
		smooths[i] = 1 - smoothness

	v1 = spline.points[0] - spline.points[-1]
	v2 = spline.points[1] - spline.points[0]
	angle = np.arccos(np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))))
	smoothness = np.abs(angle - (np.pi / 2)) / (np.pi / 2)
	smooths[0] = 1 - smoothness

	v1 = spline.points[-1] - spline.points[-2]
	v2 = spline.points[0] - spline.points[-1]
	angle = np.arccos(np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))))
	smoothness = np.abs(angle - (np.pi / 2)) / (np.pi / 2)
	smooths[-1] = 1 - smoothness

	return np.mean(smooths)

def distance_regularization_penalty(spline):
	dists = []
	for i in range(len(spline.points) - 1):
		dists.append(np.linalg.norm(spline.points[i] - spline.points[i+1]))
	dists.append(np.linalg.norm(spline.points[-1] - spline.points[0]))
	dists = np.array(dists, dtype=float)
	dists[dists > 1] = 1
	return np.sum(dists)

# Gradient Descent
learning_rate = 50
grad_eps = 2
max_iters = 50
stopping_thresh = 0.005
iter_num = 0
current_spline = None
try:
	while True:
		iter_num = iter_num + 1
		print "Iteration %d" % iter_num

		# Compute gradient
		grad = np.zeros(control_points.shape)
		for i in range(grad.shape[0]):
			for j in range(grad.shape[1]):
				c1 = control_points
				c2 = control_points.copy()
				c3 = control_points.copy()
				c2[i,j] = c2[i,j] + grad_eps
				c3[i,j] = c3[i,j] - grad_eps
				if current_spline is None:
					s1 = CatmullRomSpline(c1)
				else:
					s1 = current_spline
				s2 = CatmullRomSpline(c2)
				s3 = CatmullRomSpline(c3)
				temp1 = (loss(s2) - loss(s1)) / grad_eps
				temp2 = (loss(s1) - loss(s3)) / grad_eps
				grad[i,j] = (temp1 + temp2) / 2
		print grad
		print np.linalg.norm(grad, ord="fro")

		# Update control_points
		control_points = control_points + (learning_rate * reduce_learning_rate**(iter_num - 1) * grad)
		current_spline = CatmullRomSpline(control_points)

		# Draw
		plt.cla()
		Tvals = np.linspace(0, 1, 100).reshape(-1, 1)
		edge_points = current_spline(Tvals)
		plt.imshow(mask)
		plt.plot(edge_points[:,0], edge_points[:,1])
		plt.scatter(control_points[:,0], control_points[:,1])
		plt.scatter(current_spline.interior_points[:,0], current_spline.interior_points[:,1])
		plt.draw()
		plt.pause(0.001)
		plt.savefig("iteration%03d_pre.png" % iter_num)

		if move_point:
			# Remove weakest control point
			smallest_diff = None
			current_loss = loss(current_spline)
			best_idx = None
			splines = []
			for i in range(len(control_points)):
				modified_points = control_points[np.arange(len(control_points)) != i]
				modified_spline = CatmullRomSpline(modified_points)
				splines.append(modified_spline)
				diff = np.abs(loss(modified_spline) - current_loss)
				if smallest_diff is None:
					smallest_diff = diff
					best_idx = i
				elif smallest_diff > diff:
					smallest_diff = diff
					best_idx = i
			if deterministic_move_point:
				dists = []
				for i in range(len(splines[best_idx].points) - 1):
					dists.append(np.linalg.norm(splines[best_idx].points[i] - splines[best_idx].points[i+1]))
				dists.append(np.linalg.norm(splines[best_idx].points[-1] - splines[best_idx].points[0]))
				dists = np.array(dists, dtype=float)
				chunk = np.argmax(dists)
			else:
				valid_choices = []
				for i in range(len(splines[best_idx].points) - 1):
					if np.linalg.norm(splines[best_idx].points[i] - splines[best_idx].points[i+1]) > 1:
						valid_choices.append(i)
				if np.linalg.norm(splines[best_idx].points[0] - splines[best_idx].points[-1] > 1):
					valid_choices.append(len(splines[best_idx].points))
				while True:
					chunk = np.random.randint(len(splines[best_idx].points))
					if chunk in valid_choices:
						break
			new_point = splines[best_idx].segments[chunk](0.5)
			control_points = np.insert(splines[best_idx].points, chunk + 1, new_point, axis=0)
			current_spline = CatmullRomSpline(control_points)

			# Draw
			plt.cla()
			Tvals = np.linspace(0, 1, 100).reshape(-1, 1)
			edge_points = current_spline(Tvals)
			plt.imshow(mask)
			plt.plot(edge_points[:,0], edge_points[:,1])
			plt.scatter(control_points[:,0], control_points[:,1])
			plt.scatter(current_spline.interior_points[:,0], current_spline.interior_points[:,1])
			plt.draw()
			plt.pause(0.001)
			plt.savefig("iteration%03d_post.png" % iter_num)

		if iter_num == max_iters or np.linalg.norm(grad, ord="fro") < stopping_thresh:
			break
except KeyboardInterrupt:
	pass

import os
if move_point:
	os.system('ffmpeg -f image2 -r 1/0.5 -i iteration\%03d_post.png -c:v libx264 -pix_fmt yuv420p out.mp4')
else:
	os.system('ffmpeg -f image2 -r 1/0.5 -i iteration\%03d_pre.png -c:v libx264 -pix_fmt yuv420p out.mp4')