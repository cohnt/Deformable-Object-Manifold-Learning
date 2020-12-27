import time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

# Load the data
with open("data/rope_3d_dataset.npy", "rb") as f:
	data = np.load(f)

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

x_lims = (np.min(data_rotated[:,:,0]), np.max(data_rotated[:,:,0]))
y_lims = (np.min(data_rotated[:,:,1]), np.max(data_rotated[:,:,1]))
z_lims = (np.min(data_rotated[:,:,2]), np.max(data_rotated[:,:,2]))

plt.ion()

fig = plt.figure()
ax = p3.Axes3D(fig)

line = ax.plot(data_rotated[1,:,0], data_rotated[1,:,1], data_rotated[1,:,2])

def set_lims():
	ax.set_xlim(x_lims)
	ax.set_ylim(y_lims)
	ax.set_zlim(z_lims)

set_lims()
fig.canvas.draw()
plt.show(block=False)

for i in range(2, data_rotated.shape[0]):

	ax.clear()
	line = ax.plot(data_rotated[i,:,0], data_rotated[i,:,1], data_rotated[i,:,2])

	set_lims()
	fig.canvas.draw()
	time.sleep(0.01)
