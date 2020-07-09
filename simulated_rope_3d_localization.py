import time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

n_train = 601

# Load the data
with open("data/rope_3d_dataset.npy", "rb") as f:
	data = np.load(f)

# data = data[:,[0, 10, 20, 30, 40, 47],:]
# data = data[:,[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 47],:]

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

# fig = plt.figure()
# ax0 = fig.add_subplot(1, 2, 1)
# ax1 = fig.add_subplot(1, 2, 2, projection="3d")
# axes = [ax0, ax1]

# points = axes[0].scatter(embedding[:,0], embedding[:,1], c="grey", s=20**2)
# xlim = axes[0].get_xlim()
# ylim = axes[0].get_ylim()


from scipy.spatial import Delaunay

interpolator = Delaunay(embedding, qhull_options="QJ")

################
# Display Plot #
################

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
# 		mult_vec = np.zeros(len(train))
# 		mult_vec[simplex_indices] = b
# 		curve = np.sum(np.matmul(np.diag(mult_vec), train), axis=0).reshape(-1,3)
# 		# print "curve", curve
# 		axes[1].clear()
# 		axes[1].plot(curve[:,0], curve[:,1], curve[:,2])
# 		axes[1].set_xlim(mfd_xlims)
# 		axes[1].set_ylim(mfd_ylims)
# 		axes[1].set_zlim(mfd_zlims)

# 		fig.canvas.draw_idle()

# fig.canvas.mpl_connect('motion_notify_event', hover)
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()

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

# Interesting frames: 325, 400, 499, 600
frame = 600
num_points_to_track = len(data[frame])
# x_min = int(np.floor(np.min(data[frame,:,0])))
# x_max = int(np.ceil(np.max(data[frame,:,0])))
# y_min = int(np.floor(np.min(data[frame,:,1])))
# y_max = int(np.ceil(np.max(data[frame,:,1])))
# z_min = int(np.floor(np.min(data[frame,:,2])))
# z_max = int(np.ceil(np.max(data[frame,:,2])))
x_min = y_min = z_min = int(np.floor(np.min(data[0:n_train])))
x_max = y_max = z_max = int(np.ceil(np.max(data[0:n_train])))

print x_min, x_max
print data[frame]
# print y_min, y_max
# print z_min, z_max

heatmap_resolution = 0.1
heatmap_n_decimals = int(-np.log10(heatmap_resolution))
zero_index = -np.array([x_min/heatmap_resolution, y_min/heatmap_resolution, z_min/heatmap_resolution], dtype=int)
heatmap_shape = (int((x_max-x_min)/heatmap_resolution)+1, int((y_max-y_min)/heatmap_resolution)+1, int((z_max-z_min)/heatmap_resolution)+1)

from scipy.stats import special_ortho_group

class Particle():
	def __init__(self, xyz=None, orien=None, deformation=None):
		if xyz is None:
			self.xyz = (np.random.uniform(x_min, x_max),
			            np.random.uniform(y_min, y_max),
			            np.random.uniform(z_min, z_max))
			# self.xyz = data[frame,0]
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

		self.num_points = num_points_to_track
		self.compute_points()

		self.raw_weight = None
		self.normalized_weight = None

	def compute_points(self):
		raw_points = compute_deformation(interpolator, self.deformation)
		rotated_points = np.matmul(self.orien, raw_points.T)
		self.points = rotated_points + np.asarray(self.xyz).reshape(-1, 1)

	def compute_raw_weight(self, heatmap):
		running_total = 0.0
		for i in range(self.num_points):
			point = self.points[:,i]
			heatmap_coords = np.round((point - np.array([x_min, y_min, z_min])) / heatmap_resolution)
			heatmap_index = np.asarray(heatmap_coords, dtype=int)

			if (0 <= heatmap_index).all() and (heatmap_index < heatmap_shape).all():
				running_total += heatmap[tuple(heatmap_index)]

		self.raw_weight = running_total
		return self.raw_weight

print "Making heatmap"

# Used for frame 499
# part1 = data[frame, data[frame,:,0] < -2]
# part2 = data[frame, np.logical_and(data[frame,:,0] >= -2, data[frame,:,0] <= -1.5)]
# part3 = data[frame, data[frame,:,0] > -1.5]
# occluded = np.append(part1, part3, axis=0)

# Used for frame 400
# part1 = data[frame, data[frame,:,2] > -0]
# part2 = data[frame, np.logical_and(data[frame,:,2] <= -0, data[frame,:,2] >= -1)]
# part3 = data[frame, data[frame,:,2] < -1]
# occluded = np.append(part1, part3, axis=0)

# Used for frame 600
part1 = data[frame, data[frame,:,2] > 0][:19]
part2 = data[frame, data[frame,:,2] <= 0]
part3 = data[frame, data[frame,:,2] > 0][19:]
occluded = np.append(part1, part3, axis=0)

heatmap = np.zeros(heatmap_shape)
for i in range(heatmap_shape[0]):
	for j in range(heatmap_shape[1]):
		for k in range(heatmap_shape[2]):
			x = x_min + (i * heatmap_resolution)
			y = y_min + (j * heatmap_resolution)
			z = z_min + (k * heatmap_resolution)
			dists = np.linalg.norm(occluded - np.array([x, y, z]), axis=1)**2
			heatmap[i, j, k] = 1 / (1 + 10*np.min(dists))
# Verify that the heatmap is good
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot(data[frame,:,0], data[frame,:,1], data[frame,:,2])
# ax.set_xlim((x_min, x_max))
# ax.set_ylim((y_min, y_max))
# ax.set_zlim((z_min, z_max))

# points = []
# for i in range(heatmap_shape[0]):
# 	for j in range(heatmap_shape[1]):
# 		for k in range(heatmap_shape[2]):
# 			if i % 5 == 0 and j % 5 == 0 and k % 5 == 0:
# 				if heatmap[i,j,k] > 0.9:
# 					x = x_min + (i * heatmap_resolution)
# 					y = y_min + (j * heatmap_resolution)
# 					z = z_min + (k * heatmap_resolution)
# 					points.append([x, y, z])
# points = np.array(points)
# ax.scatter(points[:,0], points[:,1], points[:,2])

# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()

# try:
# 	while(True):
# 		for angle in np.arange(0, 360, 10):
# 			ax.view_init(30, angle)
# 			plt.draw()
# 			plt.pause(.1)
# except:
# 	pass

# for angle in np.arange(0, 720, 10):
# 	ax.view_init(30, angle)
# 	plt.draw()
# 	plt.pause(.1)
# plt.close(fig)

num_particles = 2000
exploration_factor = 0.1
particles = [Particle() for i in range(num_particles)]
iter_num = 0

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

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

while True:
	iter_num = iter_num + 1
	print "Iteration %d" % iter_num

	# Weight particles
	normalization_factor = 0
	weights = []
	for p in particles:
		w = p.compute_raw_weight(heatmap)
		weights.append(w)
		normalization_factor = normalization_factor + w
	weights = np.asarray(weights)
	# min_weight = np.min(weights[weights > 0])
	normalized_weights = []
	for p in particles:
		# w = (p.raw_weight - min_weight) / (max_weight - min_weight)
		w = p.raw_weight / normalization_factor
		p.normalized_weight = w
		normalized_weights.append(w)
	max_normalized_weight = np.max(normalized_weights)
	max_normalized_weight_ind = np.argmax(normalized_weights)

	if iter_num > -1:
		ax.clear()
		for p in particles:
			if p.raw_weight > -1:
				ax.plot(p.points.T[:,0], p.points.T[:,1], p.points.T[:,2], c=plt.cm.cool(p.normalized_weight / max_normalized_weight), linewidth=1)
		ax.plot(part1[:,0], part1[:,1], part1[:,2], color="black", linewidth=5)
		ax.plot(part2[:,0], part2[:,1], part2[:,2], color="black", linewidth=5, linestyle='dotted')
		ax.plot(part3[:,0], part3[:,1], part3[:,2], color="black", linewidth=5)
		ax.plot(particles[max_normalized_weight_ind].points.T[:,0], particles[max_normalized_weight_ind].points.T[:,1], particles[max_normalized_weight_ind].points.T[:,2], color="red", linewidth=3)

		ax.set_xlim(x_min, x_max)
		ax.set_ylim(y_min, y_max)
		ax.set_zlim(z_min, z_max)
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_zlabel('Z axis')
		plt.draw()
		plt.pause(0.001)
		plt.savefig("iteration%02d.png" % iter_num)

		if iter_num >= 50:
			for idx, angle in enumerate(np.arange(-60+10, 300+360, 10)):
				ax.view_init(30, angle)
				plt.draw()
				plt.pause(.001)
				plt.savefig("iteration%02d.png" % (iter_num + idx + 1))
			break
		# plt.close(fig)
		# plt.show()

	# Resample
	newParticles = []
	cs = np.cumsum(normalized_weights)
	step = 1/float((num_particles * (1-exploration_factor))+1)
	chkVal = step
	chkIdx = 0
	newParticles.append(particles[max_normalized_weight_ind])
	for i in range(1, int(np.ceil(num_particles * (1-exploration_factor)))):
		while cs[chkIdx] < chkVal:
			chkIdx = chkIdx + 1
		chkVal = chkVal + step
		newParticles.append(Particle(xyz=particles[chkIdx].xyz,
		                             orien=particles[chkIdx].orien,
		                             deformation=particles[chkIdx].deformation))
	for i in range(len(newParticles), num_particles):
		newParticles.append(Particle())

	# Add noise
	particles = newParticles
	for i in range(1, len(particles)):
		p = particles[i]
		xyz_var = 0.1
		p.xyz = p.xyz + np.random.multivariate_normal(np.zeros(3), xyz_var*np.eye(3))

		orien_var = 5
		rot = random_small_rotation(3, orien_var)
		p.orien = np.matmul(rot, p.orien)

		deformation_var = 0.1
		while True:
			delta = np.random.multivariate_normal(np.array([0, 0]), np.matrix([[deformation_var, 0], [0, deformation_var]]))
			if interpolator.find_simplex(p.deformation + delta) != -1:
				p.deformation = p.deformation + delta
				break

		p.compute_points()

import os
os.system('ffmpeg -f image2 -r 1/0.1 -i iteration\%02d.png -c:v libx264 -pix_fmt yuv420p out.mp4')