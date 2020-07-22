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

class CatmulRomSplineSegment():
	def __init__(self, P0, P1, P2, P3, alpha=0.5):
		self.P0 = P0
		self.P1 = P1
		self.P2 = P2
		self.P3 = P3
		self.alpha = alpha

		def tj(ti, Pi, Pj):
			xi, yi = Pi
			xj, yj = Pj
			return ((xj-xi)**2 + (yj-yi)**2)**alpha + ti

		self.t0 = 0
		self.t1 = tj(self.t0, self.P0, self.P1)
		self.t2 = tj(self.t1, self.P1, self.P2)
		self.t3 = tj(self.t2, self.P2, self.P3)

	def __call__(self, T):
		# T should be in [0, 1], where 0 maps to P1 and 1 maps to P2
		t = ((self.t2 - self.t1) * T) + self.t1
		A1 = (self.t1-t)/(self.t1-self.t0)*self.P0 + (t-self.t0)/(self.t1-self.t0)*self.P1
		A2 = (self.t2-t)/(self.t2-self.t1)*self.P1 + (t-self.t1)/(self.t2-self.t1)*self.P2
		A3 = (self.t3-t)/(self.t3-self.t2)*self.P2 + (t-self.t2)/(self.t3-self.t2)*self.P3
		B1 = (self.t2-t)/(self.t2-self.t0)*A1 + (t-self.t0)/(self.t2-self.t0)*A2
		B2 = (self.t3-t)/(self.t3-self.t1)*A2 + (t-self.t1)/(self.t3-self.t1)*A3
		C = (self.t2-t)/(self.t2-self.t1)*B1 + (t-self.t1)/(self.t2-self.t1)*B2
		return C

class CatmulRomSpline():
	def __init__(self, points):
		self.segments = []
		self.segments.append(CatmulRomSplineSegment(points[-1], points[0], points[1], points[2]))
		for i in range(len(points) - 3):
			self.segments.append(CatmulRomSplineSegment(points[i], points[i+1], points[i+2], points[i+3]))
		self.segments.append(CatmulRomSplineSegment(points[-3], points[-2], points[-1], points[0]))
		self.segments.append(CatmulRomSplineSegment(points[-2], points[-1], points[0], points[1]))