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

centroid = np.array(center_of_mass(mask))
