import numpy as np
import matplotlib.pyplot as plt

import coordinate_chart
import particle_filter
import visualization
import normalization
import utility

import data.mouse_dataset.mouse_dataset as mouse_dataset

#########################
# Experiment parameters #
#########################

# General parameters
track = False  # If true, track normally. If false, don't increase the frame number with each iteration.
               # False allows us to test only localizing in a single frame.

# Dataset parameters
n_train = 500        # Number of training samples to use
random_train = False # Optionally randomly select the training images from the whole dataset
test_start_ind = 0   # Can start the test sequence at a different index if desired

# Manifold learning
target_dim = 2   # The target dimension for ISOMAP.
neighbors_k = 12 # The number of neighbors used for ISOMAP.

# Particle filter
n_particles = 200         # Number of particles
exploration_factor = 0.25 # Fraction of particles used to explore
xy_var = 100              # Variance of diffusion noise added to particles' position component
theta_var = np.pi/32      # Variance of diffusion noise added to particles' orientation component
deformation_var = 250     # Variance of diffusion noise added to particles' deformation component
keep_best = True          # Keep the best guess unchanged