import numpy as np
import matplotlib.pyplot as plt

def maximize_window():
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())

def combine_images_to_video(fname_format="iteration\%03d.png"):
	# fname_format should be something the command line tool ffmpeg can understand
	# For example, iteration\%03d.png expects images with filenames ieration001.png, iteration002.png, etc.
	os.system('ffmpeg -f image2 -r 1/0.1 -i %s -c:v libx264 -pix_fmt yuv420p out.mp4' % fname_format)

def create_interactive_embedding_visulization(cc, point_cloud_dim):
	fig = plt.figure()
	ax0 = fig.add_subplot(1, 2, 1)
	ax1 = fig.add_subplot(1, 2, 2)
	axes = [ax0, ax1]

	points = axes[0].scatter(cc.embedding[:,0], cc.embedding[:,1], c="grey", s=20**2)
	xlim = axes[0].get_xlim()
	ylim = axes[0].get_ylim()


	mfd_xlims = (np.min(cc.train_data.reshape(len(cc.train_data),-1,point_cloud_dim)[:,:,0]), np.max(cc.train_data.reshape(len(cc.train_data),-1,point_cloud_dim)[:,:,0]))
	mfd_ylims = (np.min(cc.train_data.reshape(len(cc.train_data),-1,point_cloud_dim)[:,:,1]), np.max(cc.train_data.reshape(len(cc.train_data),-1,point_cloud_dim)[:,:,1]))

	def hover(event):
		xy = np.array([event.xdata, event.ydata])

		# Check if xy is in the convex hull, and get simplex indices
		simplex_num = cc.tri.find_simplex(xy)
		if simplex_num == -1:
			return
		simplex_indices = cc.tri.simplices[simplex_num]
		simplex = cc.tri.points[simplex_indices]

		# Display the simplex vertices
		axes[0].clear()
		axes[0].scatter(cc.embedding[:,0], cc.embedding[:,1], c="grey", s=20**2) # Draw all embedding points
		axes[0].scatter(cc.embedding[simplex_indices,0], cc.embedding[simplex_indices,1], c="blue", s=20**2) # Highlight the simplex points
		axes[0].plot(cc.embedding[simplex_indices[[0,1]],0], cc.embedding[simplex_indices[[0,1]],1], c="blue", linewidth=3) # Draw the simplex
		axes[0].plot(cc.embedding[simplex_indices[[1,2]],0], cc.embedding[simplex_indices[[1,2]],1], c="blue", linewidth=3)
		axes[0].plot(cc.embedding[simplex_indices[[0,2]],0], cc.embedding[simplex_indices[[0,2]],1], c="blue", linewidth=3)
		axes[0].set_xlim(xlim) # Fix axes limits
		axes[0].set_ylim(ylim)

		# Transform the selected point
		point = cc.single_inverse_mapping(xy)
		point_cloud = point.reshape(-1, point_cloud_dim)

		# Draw the original point cloud
		axes[1].clear()
		axes[1].scatter(point_cloud[:,0], point_cloud[:,1], s=20**2)
		axes[1].set_xlim(mfd_xlims)
		axes[1].set_ylim(mfd_ylims)

		# Update the figure
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect('motion_notify_event', hover)
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.show()