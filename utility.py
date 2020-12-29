import numpy as np
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt

def random_small_rotation(dimension, variance=None):
	if variance is None:
		variance = 0.05 * dimension * 180.0 / np.pi
	theta = np.random.normal(0, variance) * np.pi / 180.0
	rotMat = np.eye(dimension)
	rotMat[0,0] = np.cos(theta)
	rotMat[0,1] = -np.sin(theta)
	rotMat[1,0] = np.sin(theta)
	rotMat[1,1] = np.cos(theta)
	basis = special_ortho_group.rvs(dimension)
	basis_inv = basis.transpose()
	return basis.dot(rotMat).dot(basis_inv)

def rad2deg(r):
	return r * 180 / np.pi

def plt_maximize_window():
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())