import numpy as np

t = np.linspace(0, np.pi, 100)
data = np.array([0.5 + (0.5 * np.cos(t)), 0.5 * np.sin(t)]).transpose()

actual = np.array([0.5, 0.5])

from sklearn.manifold import Isomap
ism = Isomap(n_neighbors=5, n_components=1)
embedding = ism.fit_transform(data)

from scipy.stats import multivariate_normal
def likelihood(point):
	return multivariate_normal.pdf(point, mean=actual, cov=np.eye(2))