import time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

# Load the data
with open("data/rope_3d_dataset.npy", "rb") as f:
	data = np.load(f)

data_normalized = data[:,:,:] - np.repeat(data[:,0,:].reshape(data.shape[0], 1, data.shape[2]), data.shape[1], axis=1)

x_lims = (np.min(data[:,:,0]), np.max(data[:,:,0]))
y_lims = (np.min(data[:,:,1]), np.max(data[:,:,1]))
z_lims = (np.min(data[:,:,2]), np.max(data[:,:,2]))

plt.ion()

fig = plt.figure()
ax = p3.Axes3D(fig)

line = ax.plot(data_normalized[1,:,0], data_normalized[1,:,1], data_normalized[1,:,2])

def set_lims():
	ax.set_xlim(x_lims)
	ax.set_ylim(y_lims)
	ax.set_zlim(z_lims)

set_lims()
fig.canvas.draw()
plt.show(block=False)

for i in range(2, data_normalized.shape[0]):

	ax.clear()
	line = ax.plot(data_normalized[i,:,0], data_normalized[i,:,1], data_normalized[i,:,2])

	set_lims()
	fig.canvas.draw()
	time.sleep(0.01)
