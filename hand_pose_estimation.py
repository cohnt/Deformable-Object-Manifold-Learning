import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

print "Loading dataset..."

train_data_dir = os.path.join(os.getcwd(), "data", "nyu_hand_dataset", "train")
test_data_dir = os.path.join(os.getcwd(), "data", "nyu_hand_dataset", "test")

train_joints_fname = os.path.join(train_data_dir, "joint_data.mat")
test_joints_fname = os.path.join(train_data_dir, "joint_data.mat")

train_joints_mat = sio.loadmat(train_joints_fname)
test_joints_mat = sio.loadmat(test_joints_fname)

train_joints = train_joints_mat["joint_uvd"]
test_joints = test_joints_mat["joint_uvd"]

# Only use the first kinect
train_joints = train_joints[0]
test_joints = test_joints[0]

print "Train shape", train_joints.shape
print "Test shape", test_joints.shape