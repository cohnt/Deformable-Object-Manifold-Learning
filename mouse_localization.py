import time
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

# Dataset Parameters
n_train = 4751
n_test = 6630
filepath = "./data/mouse_dataset/"

# Camera Parameters
d1, d2 = 500, 1000
focal = -533
cx, cy = 320, 240

# Camera Projection
def xyz2uvd(jnt):
	if jnt.ndim == 2:
		u = jnt[:,0] / jnt[:,2] * focal + cx
		v = -jnt[:,1] / jnt[:,2] * focal + cy
		z = -jnt[:,2]
		return np.concatenate((u,v,z)).reshape(3,-1).T
	if jnt.ndim == 1:
		u = jnt[0] / jnt[2] * focal + cx
		v = -jnt[1] / jnt[2] * focal + cy
		z = -jnt[2]
		return np.array([u,v,z])

# Load training data
depths = []
train = []
for i in range(n_train):
	print filepath + ("train/%05i.png" % (i+1))
	depth = np.array(PIL.Image.open(filepath + ("train/%05i.png" % (i+1))))
	xyz = np.loadtxt(filepath + ("train/%05i.txt" % (i+1)))
	uvd = xyz2uvd(xyz)