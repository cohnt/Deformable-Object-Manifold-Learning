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
			fig, axes = plt.subplots(1, 2)
			axes[0].imshow(frame)
			axes[1].scatter(x_coords, y_coords)
			axes[1].set_xlim((0, 1920))
			axes[1].set_ylim((0, 1080))
			plt.show()
	else:
		break

cap.release()
