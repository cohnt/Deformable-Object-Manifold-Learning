import numpy as np
import matplotlib.pyplot as plt

def maximize_window():
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())

def combine_images_to_video(fname_format="iteration\%03d.png"):
	os.system('ffmpeg -f image2 -r 1/0.1 -i %s -c:v libx264 -pix_fmt yuv420p out.mp4' % fname_format)