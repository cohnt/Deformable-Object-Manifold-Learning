import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass

img_size = (100, 100)
mask = np.zeros(img_size)
blob = np.array([
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
	[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
	[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
])
blob_size = blob.shape
mask[10:10+blob_size[0], 10:10+blob_size[1]] = blob

spline_n_points = 8
centroid = np.array(center_of_mass(mask))
angles = np.linspace(0, 2*np.pi, spline_n_points+1)[0:-1]
control_points = np.vstack((np.cos(angles), np.sin(angles))).T

class CatmullRomSplineSegment():
	def __init__(self, P0, P1, P2, P3, alpha=0.5):
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

	def __call__(self, T):
		T = np.asarray(T).reshape(-1, 1)
		overall_t = T * self.total_length
		segment_idx = np.argmin(self.t_benchmarks < overall_t, axis=1)
		local_t = (overall_t - self.t_benchmarks[segment_idx-1].reshape(-1, 1)) / self.t_lengths[segment_idx].reshape(-1, 1)
		output = np.array([segment(t) for segment, t, in zip(self.segments[segment_idx], local_t)])
		return output.reshape(-1, 2)

	def rasterize(self, t_resolution=1000):
		Tvals = np.linspace(0, 1, t_resolution)
		points = np.rint(self(Tvals)).astype(int)
		filtered_points = np.unique(points, axis=0)
		# min_x = np.min(points[:,0])
		# max_x = np.max(points[:,0])
		# for x in range(min_x, max_x+1):
		# 	y_points = np.unique(points[points[:,0] == x], axis=0)
		return filtered_points


# control_points = np.array([[0, 1], [1, 0], [2, 0], [3, 1]])
# cms = CatmullRomSplineSegment(control_points[0], control_points[1], control_points[2], control_points[3])
# Tvals = np.linspace(0, 1, 100).reshape(-1, 1)
# points = cms(Tvals)
# plt.plot(points[:,0], points[:,1])
# plt.scatter(control_points[:,0], control_points[:,1])
# plt.show()

control_points = np.array([[0, 0], [10, 0], [20, 10], [10, 20], [0, 10]])
control_points = control_points + np.array([10, 10])
cms = CatmullRomSpline(control_points)

# for i in range(len(control_points)):
# 	Tvals = np.linspace(0, 1, 100)
# 	points = cms.segments[i](Tvals)
# 	plt.plot(points[:,0], points[:,1])
# 	plt.scatter(control_points[:,0], control_points[:,1])
# 	plt.show()

Tvals = np.linspace(0, 1, 100).reshape(-1, 1)
points = cms(Tvals)
plt.plot(points[:,0], points[:,1])
plt.scatter(control_points[:,0], control_points[:,1])
points = cms.rasterize()
plt.scatter(points[:,0], points[:,1])
plt.show()
