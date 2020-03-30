import numpy as np
import cv2
import matplotlib.pyplot as plt

num_points_to_track = 10
x_coord_start = 1920 / 4
x_coord_stop = (1920 / 4) * 2.5

manifold_data = []

cap = cv2.VideoCapture("data/colored_pencil.mp4")
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
		frame_num = frame_num + 1
		
		x_coords = np.linspace(x_coord_start, x_coord_stop, num=num_points_to_track, endpoint=True, dtype=int)
		y_coords = np.array([getRedHeight(frame, x) for x in x_coords])
		manifold_data.append(y_coords)

		if (frame_num - 1) % 10 == 0:
			fig, axes = plt.subplots(2, 1)
			frame_color_corrected = np.copy(frame)
			frame_color_corrected[:,:,[0,1,2]] = frame[:,:,[2,1,0]]
			axes[0].imshow(frame_color_corrected)
			axes[1].scatter(x_coords, 1080-y_coords)
			axes[1].set_xlim((0, 1920))
			axes[1].set_ylim((0, 1080))
			axes[1].set_aspect("equal")
			plt.show()
	else:
		break

cap.release()

from sklearn.manifold import Isomap

embedding = Isomap(n_neighbors=8, n_components=1).fit_transform(manifold_data)
print len(manifold_data)
print len(embedding)

fig, ax = plt.subplots()
ax.scatter(range(len(embedding)), embedding)
plt.show()