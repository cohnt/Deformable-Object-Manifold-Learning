import numpy as np
import cv2

num_points_to_track = 10
x_coord_start = 1920 / 4
x_coord_stop = (1920 / 4) * 3

manifold_data = []

cap = cv2.VideoCapture("data/colored_pencil.mp4")
if not cap.isOpened():
	print "Error opening video stream or file"

def getRedHeight(image, x):
	return 0

while cap.isOpened():
	ret, frame = cap.read()
	if ret:
		x_coords = np.linspace(x_coord_start, x_coord_stop, num=num_points_to_track, endpoint=True, dtype=int)
		y_coords = np.array([getRedHeight(frame, x) for x in x_coords])
		manifold_data.append(y_coords)
	else:
		break

cap.release()