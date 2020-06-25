import time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

n_train = 1000

# Load the data
with open("data/rope_3d_dataset.npy", "rb") as f:
	data = np.load(f)

data_centered = data[:,:,:] - np.repeat(data[:,0,:].reshape(data.shape[0], 1, data.shape[2]), data.shape[1], axis=1)

data_rotated = np.zeros(data_centered.shape)
for i in range(len(data_centered)):
	# https://math.stackexchange.com/a/476311
	a = data_centered[i,-1,:] / np.linalg.norm(data_centered[i,-1,:])
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
	for j in range(len(data_centered[i])):
		data_rotated[i,j,:] = np.matmul(R, data_centered[i,j,:])

mfd_xlims = (np.min(data_rotated[:,:,0]), np.max(data_rotated[:,:,0]))
mfd_ylims = (np.min(data_rotated[:,:,1]), np.max(data_rotated[:,:,1]))
mfd_zlims = (np.min(data_rotated[:,:,2]), np.max(data_rotated[:,:,2]))

train = data_rotated[0:n_train].reshape(n_train,-1)


from sklearn.manifold import Isomap

embedding = Isomap(n_neighbors=12, n_components=2).fit_transform(train)

fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2, projection="3d")
axes = [ax0, ax1]

points = axes[0].scatter(embedding[:,0], embedding[:,1], c="grey", s=20**2)
xlim = axes[0].get_xlim()
ylim = axes[0].get_ylim()


from scipy.spatial import Delaunay

interpolator = Delaunay(embedding, qhull_options="QJ")

################
# Display Plot #
################

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
		mult_vec = np.zeros(len(train))
		mult_vec[simplex_indices] = b
		curve = np.sum(np.matmul(np.diag(mult_vec), train), axis=0).reshape(-1,3)
		# print "curve", curve
		axes[1].clear()
		axes[1].plot(curve[:,0], curve[:,1], curve[:,2])
		axes[1].set_xlim(mfd_xlims)
		axes[1].set_ylim(mfd_ylims)
		axes[1].set_zlim(mfd_zlims)

		fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', hover)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

############
# Localize #
############

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

frame = 1005
x_min, x_max = -3, 0
y_min, y_max = 0, 2
z_min, z_max = -1, 1

heatmap_resolution = 0.01
heatmap_n_decimals = int(-np.log10(heatmap_resolution))
zero_index = np.array([x_min/heatmap_resolution, y_min/heatmap_resolution, z_min/heatmap_resolution], dtype=int)
heatmap_shape = (int((x_max-x_min)/heatmap_resolution)+1, int((y_max-y_min)/heatmap_resolution)+1, int((z_max-z_min)/heatmap_resolution)+1)

from scipy.stats import special_ortho_group

class Particle():
	def __init__(self, xyz=None, orien=None, deformation=None):
		if xyz is None:
			self.xyz = (np.random.uniform(x_min, x_max),
			            np.random.uniform(y_min, y_max),
			            np.random.uniform(z_min, z_max))
		else:
			self.xyz = xyz

		if orien is None:
			self.orien = special_ortho_group.rvs(3)
		else:
			self.orien = orien
		
		if deformation is None:
			deformation_ind = np.random.randint(0, len(train))
			self.deformation = embedding[deformation_ind]
		else:
			self.deformation = deformation

		self.compute_points()

		self.raw_weight = None
		self.normalized_weight = None

	def compute_points(self):
		raw_points = compute_deformation(interpolator, self.deformation)
		# print raw_points.shape
		# print raw_points
		rotated_points = np.matmul(self.orien, raw_points)
		self.points = rotated_points + np.asarray(self.xyz).reshape(-1, 1)
		# print self.points.T

	def compute_raw_weight(self, heatmap):
		running_total = 0.0
		for i in range(self.num_points):
			point = self.points[:,i]
			heatmap_coords = np.round(point * (10**heatmap_n_decimals)) + zero_index
			heatmap_index = np.asarray(heatmap_coords, dtype=int)

			if (0 <= heatmap_index).all() and (heatmap_index < heatmap_shape).all():
				running_total += heatmap[tuple(heatmap_index)]
			else:
				continue

		self.raw_weight = running_total
		return self.raw_weight