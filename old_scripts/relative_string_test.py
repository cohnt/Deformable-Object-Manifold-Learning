import numpy as np
import cv2
import matplotlib.pyplot as plt

num_points_to_track = 200
x_coord_start = 200
x_coord_stop = 1720

frame_list = []
manifold_data = []

show_video_images = False
disp_mode = "manifold" # Could also be "image"

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
		print "Frame %d" % frame_num

		x_coords = np.linspace(x_coord_start, x_coord_stop, num=num_points_to_track, endpoint=True, dtype=int)
		y_coords = np.array([getRedHeight(frame, x) for x in x_coords])
		offset = y_coords[0]
		y_coords = y_coords - offset
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

def hover(event):
	if points.contains(event)[0]:
		# print points.contains(event)[1]["ind"]
		idx_list = points.contains(event)[1]["ind"]
		idx = idx_list[0]

		axes[0].clear()
		axes[0].scatter(embedding[:,0], embedding[:,1], c="grey", s=20**2)
		axes[0].scatter([embedding[idx,0]], [embedding[idx,1]], c="blue", s=20**2)
		axes[0].set_xlim(xlim)
		axes[0].set_ylim(ylim)
		
		if disp_mode == "image":
			frame = frame_list[idx]
			frame_color_corrected = np.copy(frame)
			frame_color_corrected[:,:,[0,1,2]] = frame[:,:,[2,1,0]]
			axes[1].imshow(frame_color_corrected)
		elif disp_mode == "manifold":
			axes[1].clear()
			axes[1].set_ylim((mfd_min, mfd_max))
			axes[1].plot(manifold_data[idx])
		fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', hover)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()