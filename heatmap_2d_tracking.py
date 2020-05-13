import numpy as np
import cv2
import matplotlib.pyplot as plt

num_points_to_track = 200
x_coord_start = 200
x_coord_stop = 1620

frame_list = []
manifold_data = []

show_video_images = False
frames_to_train_on = 507
train_test_overlap = True

cap = cv2.VideoCapture("data/rope_two_hands.mp4")
if not cap.isOpened():
	print "Error opening video stream or file"

def getRedHeight(image, x):
	image = np.swapaxes(image, 0, 1)
	bgr_arr = image[x,:]
	r_b_arr = bgr_arr[:,2] - bgr_arr[:,0]
	r_g_arr = bgr_arr[:,2] - bgr_arr[:,1]
	arr = np.minimum(r_b_arr, r_g_arr)
	arr[arr < 0] = 0
	return np.argmax(arr)

frame_num = 0


while cap.isOpened():
	ret, frame = cap.read()
	if ret:
		frame_list.append(cv2.resize(frame, (320, 180)))
		frame_num = frame_num + 1
		if frame_num > frames_to_train_on:
			break
		print "Frame %d" % frame_num

		x_coords = np.linspace(x_coord_start, x_coord_stop, num=num_points_to_track, endpoint=True, dtype=int)
		y_coords = np.array([getRedHeight(frame, x) for x in x_coords])
		offset = y_coords[0]
		y_coords = y_coords - offset

		y_max = y_coords[-1]
		x_max = x_coord_stop - x_coord_start
		slope = float(y_max) / float(x_max)
		y_coords = y_coords - (slope * np.linspace(0, x_coord_stop - x_coord_start, num=num_points_to_track, endpoint=True, dtype=int))
		manifold_data.append(y_coords)

		if (frame_num - 1) % 10 == 0 and show_video_images:
			fig, axes = plt.subplots(2, 1)
			frame_color_corrected = np.copy(frame)
			frame_color_corrected[:,:,[0,1,2]] = frame[:,:,[2,1,0]]
			axes[0].imshow(frame_color_corrected)
			axes[1].scatter(x_coords, 1080-y_coords)
			axes[1].set_xlim((0, 1920))
			axes[1].set_ylim((0 + offset, 1080 + offset))
			axes[1].set_aspect("equal")
			plt.show()
	else:
		break

cap.release()

from sklearn.manifold import Isomap

embedding = Isomap(n_neighbors=12, n_components=2).fit_transform(manifold_data)

colors = np.array(range(len(embedding)), dtype=float)/float(len(embedding))

fig, axes = plt.subplots(1, 2)
points = axes[0].scatter(embedding[:,0], embedding[:,1], c="grey", s=20**2)
xlim = axes[0].get_xlim()
ylim = axes[0].get_ylim()

mfd_min = np.min(manifold_data)
mfd_max = np.max(manifold_data)

########################
# Set up interpolation #
########################

from scipy.spatial import Delaunay

interpolator = Delaunay(embedding, qhull_options="QJ")

################
# Display Plot #
################

def hover(event):
	# if points.contains(event)[0]:
	# 	# print points.contains(event)[1]["ind"]
	# 	idx_list = points.contains(event)[1]["ind"]
	# 	idx = idx_list[0]

	# 	axes[0].clear()
	# 	axes[0].scatter(embedding[:,0], embedding[:,1], c="grey", s=20**2)
	# 	axes[0].scatter([embedding[idx,0]], [embedding[idx,1]], c="blue", s=20**2)
	# 	axes[0].set_xlim(xlim)
	# 	axes[0].set_ylim(ylim)
		
	# 	if disp_mode == "image":
	# 		frame = frame_list[idx]
	# 		frame_color_corrected = np.copy(frame)
	# 		frame_color_corrected[:,:,[0,1,2]] = frame[:,:,[2,1,0]]
	# 		axes[1].imshow(frame_color_corrected)
	# 	elif disp_mode == "manifold":
	# 		axes[1].clear()
	# 		axes[1].set_ylim((mfd_min, mfd_max))
	# 		axes[1].plot(manifold_data[idx])
	# 	fig.canvas.draw_idle()
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
		mult_vec = np.zeros(len(manifold_data))
		mult_vec[simplex_indices] = b
		curve = np.sum(np.matmul(np.diag(mult_vec), manifold_data), axis=0)
		# print "curve", curve
		axes[1].clear()
		axes[1].set_ylim((mfd_min, mfd_max))
		axes[1].plot(curve)

		fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', hover)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

###################
# NOW WE TRACK... #
###################

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
		mult_vec = np.zeros(len(manifold_data))
		mult_vec[simplex_indices] = b
		curve = np.sum(np.matmul(np.diag(mult_vec), manifold_data), axis=0)

		points = np.vstack((np.linspace(0, x_coord_stop - x_coord_start, num=num_points_to_track, endpoint=True), curve))
		return points
	else:
		print "Error: outside of convex hull!"
		raise ValueError

red_map_list = []
cap = cv2.VideoCapture("data/rope_two_hands.mp4")
if not cap.isOpened():
	print "Error opening video stream or file"

class Particle():
	def __init__(self, xy=None, theta=None, deformation=None):
		if xy is None:
			self.xy = (np.random.randint(0, 500),
			           np.random.randint(0, 1080))
		else:
			self.xy = xy

		if theta is None:
			self.theta = np.random.rand() * (np.pi/2.0) - (np.pi/4.0)
		else:
			self.theta = theta
		
		if deformation is None:
			deformation_ind = np.random.randint(0, len(manifold_data))
			self.deformation = embedding[deformation_ind]
		else:
			self.deformation = deformation

		self.num_points = num_points_to_track
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

	def compute_raw_weight(self, red_frame):
		running_total = 0.0
		for i in range(self.num_points):
			point = self.points[:,i]
			pixel = np.asarray(np.floor(point), dtype=int)
			if pixel[0] < 0 or pixel[0] >= 1920 or pixel[1] < 0 or pixel[1] >= 1080:
				continue
			pixel = np.flip(pixel).flatten()
			running_total += red_frame[pixel[0], pixel[1]]
		self.raw_weight = running_total
		return self.raw_weight

# p = Particle()
# p.theta = 0
# p.compute_points()

# fig, ax = plt.subplots()
# ax.plot(p.points.T[:,0], p.points.T[:,1])

# p.theta = np.pi/12
# p.compute_points()

# ax.plot(p.points.T[:,0], p.points.T[:,1])
# plt.show()

noise_list = []
for _ in range(50):
	i = np.random.randint(1080)
	j = np.random.randint(1920)
	w = np.random.randint(10, 30)
	h = np.random.randint(10, 30)
	noise_list.append([i, j, w, h])
constant_noise = True

occlusion_list = []
for _ in range(25):
	i = np.random.randint(1080)
	j = np.random.randint(1920)
	w = np.random.randint(25, 50)
	h = np.random.randint(25, 100)
	occlusion_list.append([i, j, w, h])
constant_occlusion = True

num_particles = 1000
exploration_factor = 0.1
particles = [Particle() for i in range(num_particles)]
disp_thresh = 0.8
iter_num = 0
keep_mle = True
frame_num = 0
while cap.isOpened():
	ret, frame = cap.read()
	if ret:
		frame_num = frame_num + 1
		if frame_num <= frames_to_train_on and not train_test_overlap:
			continue

		iter_num = iter_num + 1

		frame_corrected = np.copy(frame)
		frame_corrected[:,:,[0,1,2]] = frame[:,:,[2,1,0]]

		# Add red
		if constant_noise:
			for noise in noise_list:
				i = noise[0]
				j = noise[1]
				w = noise[2]
				h = noise[3]
				for i1 in range(i-h, i+h):
					for j1 in range(j-w, j+w):
						try:
							frame_corrected[i1,j1] = [130, 50, 50]
						except:
							pass
		else:
			for _ in range(25):
				i = np.random.randint(frame_corrected.shape[0])
				j = np.random.randint(frame_corrected.shape[1])
				w = np.random.randint(25, 50)
				h = np.random.randint(25, 50)
				for i1 in range(i-h, i+h):
					for j1 in range(j-w, j+w):
						try:
							frame_corrected[i1,j1] = [130, 50, 50]
						except:
							pass

		# Add occlusions
		if constant_occlusion:
			for occlusion in occlusion_list:
				i = occlusion[0]
				j = occlusion[1]
				w = occlusion[2]
				h = occlusion[3]
				for i1 in range(i-h, i+h):
					for j1 in range(j-w, j+w):
						try:
							frame_corrected[i1,j1] = [255, 255, 255]
						except:
							pass
		else:
			pass

		frame_corrected = np.asarray(frame_corrected, dtype=float)
		red_matrix = np.asarray(frame_corrected[:,:,0] - np.maximum(frame_corrected[:,:,1], frame_corrected[:,:,2]), dtype=float)
		red_matrix[red_matrix < 0] = 0
		np.set_printoptions(threshold=np.inf)
		# print red_matrix[200,:]
		normalized_red_matrix = red_matrix / np.max(red_matrix)
		# print normalized_red_matrix[200,:]

		# for _ in range(25):
		# 	i = np.random.randint(normalized_red_matrix.shape[0])
		# 	j = np.random.randint(normalized_red_matrix.shape[1])
		# 	w = np.random.randint(25, 50)
		# 	h = np.random.randint(10, 25)
		# 	for i1 in range(i-w, i+w):
		# 		for j1 in range(j-h, j+h):
		# 			try:
		# 				normalized_red_matrix[i1,j1] = 1.0
		# 			except:
		# 				pass

		# fig, ax = plt.subplots()
		# ax.imshow(normalized_red_matrix, cmap="gray")
		# plt.savefig("actual_%s.png" % str(iter_num).zfill(2))
		# plt.close(fig)

		# SMOOTH IT
		from scipy.ndimage import gaussian_filter
		# print normalized_red_matrix[200,:]
		normalized_red_matrix = gaussian_filter(normalized_red_matrix, sigma=50, output=float)
		# print normalized_red_matrix[200,:]

		# fig, ax = plt.subplots()
		# ax.imshow(normalized_red_matrix, cmap="gray")
		# plt.show()

		# Weight particles
		weights = []
		for p in particles:
			weights.append(p.compute_raw_weight(normalized_red_matrix))
		weights = np.asarray(weights)
		max_weight = np.sum(weights)
		# min_weight = np.min(weights[weights > 0])
		normalized_weights = []
		for p in particles:
			# w = (p.raw_weight - min_weight) / (max_weight - min_weight)
			w = p.raw_weight / max_weight
			p.normalized_weight = w
			normalized_weights.append(w)
		max_normalized_weight = np.max(normalized_weights)
		max_normalized_weight_ind = np.argmax(normalized_weights)

		# Display
		fig, axes = plt.subplots(2, 2)
		axes[0,0].imshow(normalized_red_matrix, cmap="gray")
		axes[1,0].imshow(normalized_red_matrix, cmap="gray")
		axes[0,1].imshow(np.asarray(frame_corrected, dtype=int))
		axes[1,1].imshow(np.asarray(frame_corrected, dtype=int))

		axes[0,0].set_title("All Particles")
		axes[1,0].set_title("Good Particles")
		axes[0,1].set_title("MLE")
		axes[1,1].set_title("Mean")

		for p in particles:
			if p.normalized_weight > 0:
				axes[0,0].plot(p.points.T[:,0], p.points.T[:,1], c=plt.cm.cool(p.normalized_weight / max_normalized_weight), linewidth=1)
				if p.normalized_weight / max_normalized_weight > disp_thresh:
					axes[1,0].plot(p.points.T[:,0], p.points.T[:,1], c=plt.cm.cool(p.normalized_weight / max_normalized_weight), linewidth=2)
		p = particles[max_normalized_weight_ind]
		
		axes[0,1].plot(p.points.T[:,0], p.points.T[:,1], c="red", linewidth=3)

		x_vals = np.array([p.points.T[:,0] for p in particles]).reshape(num_particles, num_points_to_track)
		y_vals = np.array([p.points.T[:,1] for p in particles]).reshape(num_particles, num_points_to_track)
		x_avg = np.average(x_vals, axis=0, weights=normalized_weights)
		y_avg = np.average(y_vals, axis=0, weights=normalized_weights)
		axes[1,1].plot(x_avg.flatten(), y_avg.flatten(), c="red", linewidth=3)


		axes[0,0].set_xlim((0,1920))
		axes[0,0].set_ylim((1080,0))
		axes[1,0].set_xlim((0,1920))
		axes[1,0].set_ylim((1080,0))
		axes[0,1].set_xlim((0,1920))
		axes[0,1].set_ylim((1080,0))
		axes[1,1].set_xlim((0,1920))
		axes[1,1].set_ylim((1080,0))

		# mng = plt.get_current_fig_manager()
		# mng.resize(*mng.window.maxsize())
		# plt.show()
		plt.savefig("iteration_%s.svg" % str(iter_num).zfill(2))
		plt.close(fig)
		print "Saved %d" % iter_num

		# Resample
		newParticles = []
		cs = np.cumsum(normalized_weights)
		step = 1/float((num_particles * (1-exploration_factor))+1-int(keep_mle))
		chkVal = step
		chkIdx = 0
		for i in range(int(keep_mle), int(np.ceil(num_particles * (1-exploration_factor)))):
			while cs[chkIdx] < chkVal:
				chkIdx = chkIdx + 1
			chkVal = chkVal + step
			newParticles.append(Particle(xy=particles[chkIdx].xy,
			                             theta=particles[chkIdx].theta,
			                             deformation=particles[chkIdx].deformation))
		for i in range(len(newParticles), num_particles-int(keep_mle)):
			newParticles.append(Particle())

		if keep_mle:
			newParticles.append(particles[max_normalized_weight_ind])

		# Add noise
		particles = newParticles
		for p in particles:
			xy_var = 200
			p.xy = p.xy + np.random.multivariate_normal(np.array([0, 0]), np.matrix([[xy_var, 0], [0, xy_var]]))

			theta_var = np.pi/16
			p.theta = p.theta + np.random.normal(0, theta_var)
			p.theta = ((p.theta + np.pi/4.0) % (np.pi/2.0)) - np.pi/4.0

			deformation_var = 500
			while True:
				delta = np.random.multivariate_normal(np.array([0, 0]), np.matrix([[deformation_var, 0], [0, deformation_var]]))
				if interpolator.find_simplex(p.deformation + delta) != -1:
					p.deformation = p.deformation + delta
					break

			p.compute_points()
	else:
		break

cap.release()