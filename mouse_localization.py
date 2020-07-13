import time
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

# Dataset Parameters
n_train_max = 4751
n_test_max = 6630
n_tracked_points = 5
filepath = "./data/mouse_dataset/"

# Subset
n_train = 400
n_test = 100
train_inds = np.random.choice(n_train_max, n_train, replace=False)
test_inds = np.random.choice(n_test_max, n_test, replace=False)

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
train_depths = []
train_xyz = []
train_uvd = []
for i in train_inds:
	print filepath + ("train/%05i.png" % (i+1))
	depth = np.array(PIL.Image.open(filepath + ("train/%05i.png" % (i+1))))
	xyz = np.loadtxt(filepath + ("train/%05i.txt" % (i+1)))
	uvd = xyz2uvd(xyz)

	train_depths.append(depth)
	train_xyz.append(xyz)
	train_uvd.append(uvd)

train_depths = np.array(train_depths)
train_xyz = np.array(train_xyz)
train_uvd = np.array(train_uvd)

# Load testing data
test_depths = []
test_xyz = []
test_uvd = []
for i in test_inds:
	print filepath + ("test/%05i.png" % (i+1))
	depth = np.array(PIL.Image.open(filepath + ("test/%05i.png" % (i+1))))
	xyz = np.loadtxt(filepath + ("test/%05i.txt" % (i+1)))
	uvd = xyz2uvd(xyz)

	test_depths.append(depth)
	test_xyz.append(xyz)
	test_uvd.append(uvd)

test_depths = np.array(test_depths)
test_xyz = np.array(test_xyz)
test_uvd = np.array(test_uvd)

# Center the data
train_uvd_centered = train_uvd[:,:,:] - np.repeat(train_uvd[:,0,:].reshape(train_uvd.shape[0], 1, train_uvd.shape[2]), train_uvd.shape[1], axis=1)

# Fix the rotation
train_uvd_rotated = np.zeros(train_uvd_centered.shape)
for i in range(len(train_uvd_centered)):
	# https://math.stackexchange.com/a/476311
	a = train_uvd_centered[i,-1,:] / np.linalg.norm(train_uvd_centered[i,-1,:])
	b = np.array([1, 0, 0])
	v = np.cross(a, b)
	s = np.linalg.norm(v)
	c = np.dot(a, b)
	vx = np.array([
		[0, -v[2], v[1]],
		[v[2], 0, -v[0]],
		[-v[1], v[0], 0]
	])
	R = np.eye(3) + vx + np.dot(vx, vx)*(1 / (1+c))
	for j in range(len(train_uvd_centered[i])):
		train_uvd_rotated[i,j,:] = np.matmul(R, train_uvd_centered[i,j,:])
