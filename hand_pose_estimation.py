import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import os

# Dataset Parameters
xRes = 640
yRes = 480
xzFactor = 1.08836710
yzFactor = 0.817612648
halfResX = xRes/2
halfResY = yRes/2
coeffX = 588.036865
coeffY = 587.075073

print "Loading dataset..."

train_data_dir = os.path.join(os.getcwd(), "data", "nyu_hand_dataset", "train")
test_data_dir = os.path.join(os.getcwd(), "data", "nyu_hand_dataset", "test")

train_joints_fname = os.path.join(train_data_dir, "joint_data.mat")
test_joints_fname = os.path.join(test_data_dir, "joint_data.mat")

train_joints_mat = sio.loadmat(train_joints_fname)
test_joints_mat = sio.loadmat(test_joints_fname)

train_joints = train_joints_mat["joint_uvd"]
test_joints = test_joints_mat["joint_uvd"]

# Only use the first kinect
train_joints = train_joints[0]
test_joints = test_joints[0]

print "Train shape", train_joints.shape
print "Test shape", test_joints.shape

def depth_to_uvd(depth):
	V, U = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
	uvd = np.stack((U, V, depth), axis=2)
	return uvd

def uvd_to_xyz(uvd):
	normalizedX = (uvd[:,:,0] / xRes) - 0.5
	normalizedY = 0.5 - (uvd[:,:,0] / yRes)
	xyz = np.zeros(uvd.shape, dtype=float)
	xyz[:,:,2] = uvd[:,:,2]
	xyz[:,:,0] = np.multiply(normalizedX, xyz[:,:,2]) * xzFactor
	xyz[:,:,1] = np.multiply(normalizedY, xyz[:,:,2]) * yzFactor
	return xyz

def xyz_to_uvd(xyz):
	uvd = np.zeros(xyz.shape)
	uvd[:,:,0] = np.divide(coeffX * xyz[:,:,0], xyz[:,:,2] + halfResX)
	uvd[:,:,1] = np.divide(halfResY - (coeffY * xyz[:,:,1]), xyz[:,:,2])
	uvd[:,:,2] = xyz[:,:,2]
	return uvd

def parse_16_bit_depth(image):
	return image[:,:,2] + np.left_shift(np.uint16(image[:,:,1]), np.uint16(8))

print "Displaying example"
fig = plt.figure()
ax = fig.add_subplot(111)
idxes = np.random.choice(train_joints.shape[0], 10, replace=False)
for idx in idxes:
	depth_image = matplotlib._png.read_png_int(os.path.join(train_data_dir, ("depth_1_%07d.png" % (idx+1))))
	depth = parse_16_bit_depth(depth_image)
	ax.cla()
	ax.imshow(depth, cmap="gray")
	ax.scatter(train_joints[idx,:,0], train_joints[idx,:,1])
	plt.draw()
	plt.pause(1)