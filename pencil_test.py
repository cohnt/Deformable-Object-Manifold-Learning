import numpy as np
import cv2

num_points_to_track = 10
x_coord_start = 1920 / 4
x_coord_stop = (1920 / 4) * 3

cap = cv2.VideoCapture("data/colored_pencil.mp4")
if not cap.isOpened():
	print "Error opening video stream or file"

while cap.isOpened():
	ret, frame = cap.read()
	if ret:
		pass
	else:
		break

cap.release()