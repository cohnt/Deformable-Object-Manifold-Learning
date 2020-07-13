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
# test_inds = np.random.choice(n_test_max, n_test, replace=False)
test_inds = range(n_test)

# Camera Parameters
d1, d2 = 500, 1000
focal = -533
cx, cy = 320, 240

# Experiment Parameters
frame = 0
gaussian_filter_sigma = 3

# Particle Filter Parameters
n_particles = 200
exploration_factor = 0.25

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

# Compute the manifold limits (for the embedding plots)
mfd_xlims = (np.min(train_uvd_rotated[:,:,0]), np.max(train_uvd_rotated[:,:,0]))
mfd_ylims = (np.min(train_uvd_rotated[:,:,1]), np.max(train_uvd_rotated[:,:,1]))
mfd_zlims = (np.min(train_uvd_rotated[:,:,2]), np.max(train_uvd_rotated[:,:,2]))

# Compute the Isomap embedding
from sklearn.manifold import Isomap

train_uvd_flattened = train_uvd_rotated.reshape(n_train, -1)
embedding = Isomap(n_neighbors=12, n_components=2).fit_transform(train_uvd_flattened)

# fig = plt.figure()
# ax0 = fig.add_subplot(1, 2, 1)
# ax1 = fig.add_subplot(1, 2, 2)
# axes = [ax0, ax1]

# points = axes[0].scatter(embedding[:,0], embedding[:,1], c="grey", s=20**2)
# xlim = axes[0].get_xlim()
# ylim = axes[0].get_ylim()

# Compute the Delaunay triangulation
from scipy.spatial import Delaunay

interpolator = Delaunay(embedding, qhull_options="QJ")

# # Make the interactive plot
# def hover(event):
# 	xy = np.array([event.xdata, event.ydata])

# 	# Check if xy is in the convex hull
# 	simplex_num = interpolator.find_simplex(xy)
# 	# print "xy", xy, "\tsimplex_num", simplex_num
# 	if simplex_num != -1:
# 		# Get the simplex
# 		simplex_indices = interpolator.simplices[simplex_num]
# 		# print "simplex_indices", simplex_indices
# 		simplex = interpolator.points[simplex_indices]
# 		# print "simplex", simplex

# 		# Display the simplex vertices
# 		axes[0].clear()
# 		axes[0].scatter(embedding[:,0], embedding[:,1], c="grey", s=20**2)
# 		axes[0].scatter(embedding[simplex_indices,0], embedding[simplex_indices,1], c="blue", s=20**2)
# 		axes[0].plot(embedding[simplex_indices[[0,1]],0], embedding[simplex_indices[[0,1]],1], c="blue", linewidth=3)
# 		axes[0].plot(embedding[simplex_indices[[1,2]],0], embedding[simplex_indices[[1,2]],1], c="blue", linewidth=3)
# 		axes[0].plot(embedding[simplex_indices[[0,2]],0], embedding[simplex_indices[[0,2]],1], c="blue", linewidth=3)
# 		axes[0].set_xlim(xlim)
# 		axes[0].set_ylim(ylim)

# 		# Compute barycentric coordinates
# 		A = np.vstack((simplex.T, np.ones((1, 3))))
# 		b = np.vstack((xy.reshape(-1, 1), np.ones((1, 1))))
# 		b_coords = np.linalg.solve(A, b)
# 		b = np.asarray(b_coords).flatten()
# 		print "b_coords", b, np.sum(b_coords)

# 		# Interpolate the deformation
# 		mult_vec = np.zeros(len(train_uvd_flattened))
# 		mult_vec[simplex_indices] = b
# 		curve = np.sum(np.matmul(np.diag(mult_vec), train_uvd_flattened), axis=0).reshape(-1,3)
# 		# print "curve", curve
# 		axes[1].clear()
# 		axes[1].plot(curve[:,0], curve[:,1])
# 		axes[1].set_xlim(mfd_xlims)
# 		axes[1].set_ylim(mfd_ylims)

# 		fig.canvas.draw_idle()

# fig.canvas.mpl_connect('motion_notify_event', hover)
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()

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
		mult_vec = np.zeros(len(train_uvd_flattened))
		mult_vec[simplex_indices] = b
		curve = np.sum(np.matmul(np.diag(mult_vec), train_uvd_flattened), axis=0).reshape(-1,3)
		return curve[0:2] # Ignore the third dimension
	else:
		print "Error: outside of convex hull!"
		raise ValueError

x_min = y_min = int(np.floor(np.min(test_uvd)))
x_max = y_max = z_max = int(np.ceil(np.max(test_uvd)))

# Create the heatmap
heatmap = np.zeros(train_depths[0].shape)
for i in range(heatmap.shape[0]):
	for j in range(heatmap.shape[1]):
		heatmap[i,j] = 1.0 if test_depths[frame,i,j] == 1000.0 else 0.0

from scipy.ndimage import gaussian_filter
heatmap = gaussian_filter(heatmap, sigma=gaussian_filter_sigma, output=float)

# plt.imshow(heatmap, cmap=plt.cm.gray)
# plt.show()

class Particle():
	def __init__(self, xy=None, theta=None, deformation=None):
		if xy is None:
			self.xy = (np.random.randint(x_min, x_max),
			           np.random.randint(y_min, y_max))
		else:
			self.xy = xy

		if theta is None:
			self.theta = (np.random.rand() * np.pi) - (np.pi/2.0)
		else:
			self.theta = theta
		
		if deformation is None:
			deformation_ind = np.random.randint(0, len(embedding))
			self.deformation = embedding[deformation_ind]
		else:
			self.deformation = deformation

		self.n_points = n_tracked_points
		self.compute_points()

		self.raw_weight = None
		self.normalized_weight = None

	def rotation_matrix(self):
		return np.matrix([[np.cos(self.theta), -np.sin(self.theta)],
		                  [np.sin(self.theta),  np.cos(self.theta)]])

	def compute_points(self):
		raw_points = compute_deformation(interpolator, self.deformation)
		# print raw_points.shape
		# print raw_points
		rotated_points = np.matmul(self.rotation_matrix(), raw_points)
		self.points = rotated_points + np.asarray(self.xy).reshape(-1, 1)
		# print self.points.T

	def compute_raw_weight(self, heatmap):
		running_total = 0.0
		for i in range(self.n_points):
			point = self.points[:,i]
			pixel = np.asarray(np.floor(point), dtype=int)
			if pixel[0] < x_min or pixel[0] >= x_max or pixel[1] < y_min or pixel[1] >= y_max:
				continue
			pixel = np.flip(pixel).flatten()
			running_total += heatmap[pixel[0], pixel[1]]
		self.raw_weight = running_total
		return self.raw_weight

# Run the particle filter
particles = [Particle() for i in range(n_particles)]
iter_num = 0