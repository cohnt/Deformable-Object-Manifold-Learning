import numpy as np

class ParticleFilter():
	def __init__(self, dimension, n_particles, exploration_factor, keep_best, RandomSampler, Likelihood, Diffuser):
		# Particles will simply be stored as numpy arrays (of size "dimension").
		# exploration_factor is the fraction (between 0 and 1) of particles which are randomly sampled at each iteration.
		# keep_best is a boolean that determines whether or not the best particle is kept without any diffusion noise.
		# RandomSampler is a function which returns a random valid particle.
		# Likelihood is a function which takes in a particle, and returns the likelihood that the particle is the ground truth.
		# Diffuser is a function which takes in a particle, and returns a new particle with added noise.
		self.dimension = dimension
		self.n_particles = n_particles
		self.exploration_factor = exploration_factor
		self.keep_best = keep_best
		self.RandomSampler = RandomSampler
		self.Likelihood = Likelihood
		self.Diffuser = Diffuser

		self.init_particles()

	def init_particles(self):
		# Create random initial particles.
		self.particles = np.array([self.RandomSampler() for _ in range(self.n_particles)])
		self.weights = np.zeros(n_particles)
		self.max_weight_ind = -1

	def weight(self):
		# Compute weights for all particles, and normalize weights so their sum is 1.
		# Determine the highest weight particle.
		for i in range(n_particles):
			self.weights[i] = self.Likelihood(self.particles[i])
		normalizaion_factor = np.sum(self.weights)
		self.weights = self.weights / normalizaion_factor
		self.max_weight_ind = np.argmax(self.weights)

	def predict_mle(self):
		# Return highest likelihood particle
		return self.particles[self.max_weight_ind]

	def predict_mean(self):
		# Return weighted average of all particles
		return np.average(self.particles, weights=self.weights, axis=0)

	def resample(self):
		# Perform importance resampling, while keeping the best estimate if specified.
		# Also add in the specified number of random exploration particles.
		new_particles = []

		# Determine step size
		n_importance_resampling = self.n_particles * (1-exploration_factor)
		if self.keep_best:
			n_importance_resampling = n_importance_resampling - 1
			new_particles.append(self.particles[self.max_weight_ind].copy())
		step_size = 1/float(n_importance_resampling)

		# Importance resampling
		chkVal = step
		chkIdx = 0
		cs = np.cumsum(self.weights)
		for i in range(n_importance_resampling):
			# Find the next sample
			while cs[chkIdx] < chkVal:
				chkIdx = chkIdx + 1
			chkVal = chkVal + step
			new_particles.append(self.particles[chkIdx].copy())

		# Exploration particles
		while(len(new_particles) < self.n_particles):
			new_particles.append(self.RandomSampler())

		self.particles = np.array(new_particles)

	def diffuse(self):
		for i in range(self.n_particles):
			self.particles[i] = Diffuser(self.particles[i])