import matplotlib.pyplot as plt
import numpy as np

# K-Means clustering example:
# Generating 2000 samples from each centroids:
# [-0.5 -0.5]
# [   0  0.5]
# Training the model...
# Model Centroids:
# [-0.390 -0.253]
# [-0.057  0.167]
# Classifying the samples...
# Samples closest to first centroid: 1984
# Samples closest to second centroid: 2016

# These are the values used to generate the data.
ORIGINAL_CENTROIDS = np.array([[-0.5, -0.5], [0, 0.5]])
MODEL_CENTROIDS = np.array([[-0.390, -0.253],[-0.057, 0.167]])

def read_samples_in(input_file):
	return np.loadtxt(input_file, delimiter=',')

def plot_samples(input_file):
	samples = read_samples_in(input_file)

	plt.scatter(samples[:,0], samples[:,1], lw=0)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig("../k_means_samples.jpg")
	plt.show()

def plot_samples_with_original_centroids(input_file):
	samples = read_samples_in(input_file)

	plt.figure(figsize=(10,7))
	ax = plt.gca()
	plt.scatter(samples[:,0], samples[:,1], lw=0)
	plt.scatter(ORIGINAL_CENTROIDS[:,0], ORIGINAL_CENTROIDS[:,1], color='r', s=100, lw=0)
	
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig("../k_means_samples_with_original.jpg")
	plt.show()


def plot_samples_with_model_centroids(input_file):
	samples = read_samples_in(input_file)

	plt.figure(figsize=(10,7))
	ax = plt.gca()

	plt.scatter(samples[:,0], samples[:,1], c=['g' if sample == 0 else 'b' for sample in samples[:,2]], lw=0)
	plt.scatter(MODEL_CENTROIDS[:,0], MODEL_CENTROIDS[:,1], color='r', s=100, lw=0)
	
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig("../k_means_samples_with_model.jpg")
	plt.show()


if __name__ == '__main__':
	plot_samples('k_means_samples.csv')
	plot_samples_with_original_centroids('k_means_samples.csv')
	plot_samples_with_model_centroids('k_means_samples.csv')
