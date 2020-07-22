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
	def __init__(self, P0, P1, P2, P3):
		self.P0 = P0
		self.P1 = P1
		self.P2 = P2
		self.P3 = P3

class CatmulRomSpline():
	def __init__(self, points):
		self.segments = []
		self.segments.append(CatmulRomSplineSegment(points[-1], points[0], points[1], points[2]))
		for i in range(len(points) - 3):
			self.segments.append(CatmulRomSplineSegment(points[i], points[i+1], points[i+2], points[i+3]))
		self.segments.append(CatmulRomSplineSegment(points[-3], points[-2], points[-1], points[0]))
		self.segments.append(CatmulRomSplineSegment(points[-2], points[-1], points[0], points[1]))