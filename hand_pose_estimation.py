import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.io as sio
import os

from sklearn.manifold import Isomap
from scipy.spatial import Delaunay

# Dataset Parameters
xRes = 640
yRes = 480
xzFactor = 1.08836710
yzFactor = 0.817612648
halfResX = xRes/2
halfResY = yRes/2
coeffX = 588.036865
coeffY = 587.075073
n_train = 500

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

# Subset the training data
train_image_indexes = np.random.choice(train_joints.shape[0], n_train, replace=False)
train_joints = train_joints[train_image_indexes]

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

# print "Displaying example"
# fig = plt.figure()
# ax = fig.add_subplot(111)
# idxes = np.random.choice(train_image_indexes.shape[0], 10, replace=False)
# for idx in idxes:
# 	depth_image = matplotlib._png.read_png_int(os.path.join(train_data_dir, ("depth_1_%07d.png" % (train_image_indexes[idx]))))
# 	depth = parse_16_bit_depth(depth_image)
# 	ax.cla()
# 	ax.imshow(depth, cmap="gray")
# 	ax.scatter(train_joints[idx,:,0], train_joints[idx,:,1])
# 	plt.draw()
# 	plt.pause(1)

# Center and rotate the data, in preparation for manifold learning
train_uvd_centered = train_joints[:,:,:] - np.repeat(train_joints[:,0,:].reshape(train_joints.shape[0], 1, train_joints.shape[2]), train_joints.shape[1], axis=1)
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

mfd_xlims = (np.min(train_uvd_rotated[:,:,0]), np.max(train_uvd_rotated[:,:,0]))
mfd_ylims = (np.min(train_uvd_rotated[:,:,1]), np.max(train_uvd_rotated[:,:,1]))
mfd_zlims = (np.min(train_uvd_rotated[:,:,2]), np.max(train_uvd_rotated[:,:,2]))

train_uvd_flattened = train_uvd_rotated.reshape(n_train, -1)
embedding = Isomap(n_neighbors=12, n_components=2).fit_transform(train_uvd_flattened)
interpolator = Delaunay(embedding, qhull_options="QJ")

fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2, projection="3d")
axes = [ax0, ax1]

points = axes[0].scatter(embedding[:,0], embedding[:,1], c="grey", s=20**2)
xlim = axes[0].get_xlim()
ylim = axes[0].get_ylim()

# Make the interactive plot
def hover(event):
	xy = np.array([event.xdata, event.ydata])

	# Check if xy is in the convex hull
	simplex_num = interpolator.find_simplex(xy)
	# print "xy", xy, "\tsimplex_num", simplex_num
	if simplex_num != -1:
		# Get the simplex
		simplex_indices = interpolator.simplices[simplex_num]
		# print "simplex_indices", simplex_indices
		simplex = interpolator.points[simplex_indices]
		# print "simplex", simplex

		# Display the simplex vertices
		axes[0].clear()
		axes[0].scatter(embedding[:,0], embedding[:,1], c="grey", s=20**2)
		axes[0].scatter(embedding[simplex_indices,0], embedding[simplex_indices,1], c="blue", s=20**2)
		axes[0].plot(embedding[simplex_indices[[0,1]],0], embedding[simplex_indices[[0,1]],1], c="blue", linewidth=3)
		axes[0].plot(embedding[simplex_indices[[1,2]],0], embedding[simplex_indices[[1,2]],1], c="blue", linewidth=3)
		axes[0].plot(embedding[simplex_indices[[0,2]],0], embedding[simplex_indices[[0,2]],1], c="blue", linewidth=3)
		axes[0].set_xlim(xlim)
		axes[0].set_ylim(ylim)

		# Compute barycentric coordinates
		A = np.vstack((simplex.T, np.ones((1, 3))))
		b = np.vstack((xy.reshape(-1, 1), np.ones((1, 1))))
		b_coords = np.linalg.solve(A, b)
		b = np.asarray(b_coords).flatten()
		print "b_coords", b, np.sum(b_coords)

		# Interpolate the deformation
		mult_vec = np.zeros(len(train_uvd_flattened))
		mult_vec[simplex_indices] = b
		curve = np.sum(np.matmul(np.diag(mult_vec), train_uvd_flattened), axis=0).reshape(-1,3)
		# print "curve", curve
		axes[1].clear()
		axes[1].view_init(30, 225)
		axes[1].plot(curve[0:6,0], curve[0:6,1], curve[0:6,2])
		axes[1].plot(curve[6:12,0], curve[6:12,1], curve[6:12,2])
		axes[1].plot(curve[12:18,0], curve[12:18,1], curve[12:18,2])
		axes[1].plot(curve[18:24,0], curve[18:24,1], curve[18:24,2])
		axes[1].plot(curve[24:29,0], curve[24:29,1], curve[24:29,2])
		axes[1].scatter(curve[29:,0], curve[29:,1], curve[29:,2])
		axes[1].set_xlim(mfd_xlims)
		axes[1].set_ylim(mfd_ylims)
		axes[1].set_zlim(mfd_zlims)

		fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', hover)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

def compute_deformation(interpolator, deformation_coords):
	simplex_num = interpolator.find_simplex(deformation_coords)
	if simplex_num != -1:
		simplex_indices = interpolator.simplices[simplex_num]
		simplex = interpolator.points[simplex_indices]

		# Compute barycentric coordinates
		A = np.vstack((simplex.T, np.ones((1, 3))))
		b = np.vstack((deformation_coords.reshape(-1, 1), np.ones((1, 1))))
		b_coords = np.linalg.solve(A, b)
		b = np.asarray(b_coords).flatten()

		# Interpolate the deformation
		mult_vec = np.zeros(len(train))
		mult_vec[simplex_indices] = b
		curve = np.sum(np.matmul(np.diag(mult_vec), train), axis=0).reshape(-1,3)
		return curve
	else:
		print "Error: outside of convex hull!"
		raise ValueError

frame = np.random.choice(test_joints.shape[0])
num_points_to_track = len(test_joints[frame])
x_min = int(np.floor(np.min(test_joints[frame,:,0])))
y_min = int(np.floor(np.min(test_joints[frame,:,1])))
z_min = int(np.floor(np.min(test_joints[frame,:,2])))
x_max = int(np.ceil(np.max(test_joints[frame,:,0])))
y_max = int(np.ceil(np.max(test_joints[frame,:,1])))
z_max = int(np.ceil(np.max(test_joints[frame,:,2])))

print x_min, x_max
print y_min, y_max
print z_min, z_max

heatmap_resolution = 1
heatmap_n_decimals = int(-np.log10(heatmap_resolution))
zero_index = -np.array([x_min/heatmap_resolution, y_min/heatmap_resolution, z_min/heatmap_resolution], dtype=int)
heatmap_shape = (int((x_max-x_min)/heatmap_resolution)+1, int((y_max-y_min)/heatmap_resolution)+1, int((z_max-z_min)/heatmap_resolution)+1)

heatmap = np.zeros(heatmap_shape)
for i in range(heatmap_shape[0]):
	for j in range(heatmap_shape[1]):
		for k in range(heatmap_shape[2]):
			x = x_min + (i * heatmap_resolution)
			y = y_min + (j * heatmap_resolution)
			z = z_min + (k * heatmap_resolution)
			dists = np.linalg.norm(test_joints[frame] - np.array([x, y, z]), axis=1)**2
			heatmap[i, j, k] = 1 / (1 + np.min(dists))

# Verify that the heatmap is good
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(test_joints[frame,:,0], test_joints[frame,:,1], test_joints[frame,:,2])
ax.set_xlim((x_min, x_max))
ax.set_ylim((y_min, y_max))
ax.set_zlim((z_min, z_max))

points = []
for i in range(heatmap_shape[0]):
	for j in range(heatmap_shape[1]):
		for k in range(heatmap_shape[2]):
			if i % 5 == 0 and j % 5 == 0 and k % 5 == 0:
				if heatmap[i,j,k] > 0.1:
					x = x_min + (i * heatmap_resolution)
					y = y_min + (j * heatmap_resolution)
					z = z_min + (k * heatmap_resolution)
					points.append([x, y, z])
points = np.array(points)
ax.scatter(points[:,0], points[:,1], points[:,2])

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()