import numpy as np
import PIL.Image
from tqdm import tqdm

# Camera parameters
d1,d2 = 500,1000
focal = -533
cx,cy = 320,240

def xyz2uvd(jnt):
	if jnt.ndim == 2:
		u = jnt[:,0] / jnt[:,2] * focal + cx
		v = -jnt[:,1] / jnt[:,2] * focal + cy
		z = -jnt[:,2]
		return np.concatenate((u,v,z)).reshape(3,-1).T
	if jnt.ndim == 1:
		u = jnt[0] / jnt[2] * focal + cx
		v = -jnt[1] / jnt[2] * focal + cy
		z = -jnt[2]
		return np.array([u,v,z])

n_train = 4751
train_filepath = "data/mouse_dataset/train/"

n_test = 1326
test_filepath = "data/mouse_dataset/train/"

def load_train(n=n_train):
	global train_images, train_clouds, train_poses

	train_images = []
	train_clouds = []
	train_poses = []

	print "Loading train data..."
	for i in tqdm(range(n)):
		train_images.append(np.array(PIL.Image.open("%s%05i.png" % (train_filepath, i+1))))
		train_clouds.append(np.array(np.where(train_images[i] != d2)).T)
		train_poses.append(xyz2uvd(np.loadtxt("%s%05i.txt" % (train_filepath, i+1))))

def load_test(n=n_test):
	global test_images, test_clouds, test_poses

	test_images = []
	test_clouds = []
	test_poses = []

	print "Loading test data..."
	for i in tqdm(range(n)):
		test_images.append(np.array(PIL.Image.open("%s%05i.png" % (test_filepath, i+1))))
		test_clouds.append(np.array(np.where(train_images[i] != d2)).T)
		test_poses.append(xyz2uvd(np.loadtxt("%s%05i.txt" % (test_filepath, i+1))))