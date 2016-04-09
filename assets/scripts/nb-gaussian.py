import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

MEANS = [[1,1],[5,5]]
VARS = [[1,1],[0.1,0.1]]

# You'll have to believe me that these came out of the model!
MODEL_MEANS = [[1.14313, 0.9494257],[5.065183, 4.98709]]
MODEL_VARS = [[0.9174, 1.43628], [0.11356, 0.16543]]

def draw_gauss_samples(samples_per_class, means, variances):
	class_count = means.shape[0]
	features_count = means.shape[1]

	assert(class_count==variances.shape[0])
	assert(features_count==variances.shape[1])

	samples = []
	classes = []

	for i in range(class_count):
		for j in range(samples_per_class):
			classes.append(i)
			samples.append([])

			for k in range(features_count):
				samples[samples_per_class*i + j].append(np.random.normal(means[i,k], np.sqrt(variances[i,k])))

	return np.array(samples), np.array(classes)

def plot_2d_samples(samples, classes, means=None, variances=None):
	assert(samples.shape[1] == 2)

	x = samples[:,0]
	y = samples[:,1]

	plt.figure(figsize=(10,7))
	ax = plt.gca()
	plt.scatter(x,y,c=classes)

	if means is not None:
		assert(means.shape[1] == 2)

		x_mean = means[:,0]
		y_mean = means[:,0]
		plt.scatter(x_mean, y_mean, marker='o', s=150, color='black')

		if variances is not None:
			assert(variances.shape[1] == 2)

			for (v,m) in zip(variances, means):
				var_ellipse = Ellipse(xy=(m[0],m[1]), width=4*np.sqrt(v[0]), height=4*np.sqrt(v[1]), fc='none')
				ax.add_patch(var_ellipse)


	plt.xlim([-2,7])
	plt.ylim([-2,7])
	plt.title("Samples drawn from two classes of gaussian distributions")
	plt.xlabel("x")
	plt.ylabel("y")				
	plt.savefig('../gaussian_samples.png')

def plot_model_vs_true(model_means, model_vars,
						true_means, true_vars,
						samples, classes):
	assert(samples.shape[1] == 2)
	assert(model_means.shape==model_vars.shape)
	assert(true_means.shape==true_vars.shape)

	x = samples[:,0]
	y = samples[:,1]

	plt.figure(figsize=(10,7))
	ax = plt.gca()
	plt.scatter(x,y,c=classes)

	x_m_mean = model_means[:,0]
	y_m_mean = model_means[:,1]

	x_t_mean = true_means[:,0]
	y_t_mean = true_means[:,1]

	plt.scatter(x_m_mean, y_m_mean, marker='o', color='red', s=150)
	plt.scatter(x_t_mean, y_t_mean, marker='o', color='black', s=150)

	for (m,v) in zip(means, variances):
				var_ellipse = Ellipse(xy=(m[0],m[1]), width=4*np.sqrt(v[0]), height=4*np.sqrt(v[1]), fc='none')
				ax.add_patch(var_ellipse)

	for (m,v) in zip(model_means, model_vars):
				var_ellipse = Ellipse(xy=(m[0],m[1]), width=4*np.sqrt(v[0]), height=4*np.sqrt(v[1]), fc='none', ec='red')
				ax.add_patch(var_ellipse)

	plt.xlim([-2,7])
	plt.ylim([-2,7])
	plt.title("Estimated Naive Bayes distribution shown in red")
	plt.xlabel("x")
	plt.ylabel("y")				
	plt.savefig('../nb_gauss_model_estimates.png')

def save_data_to_csv(samples, classes):
	assert(len(samples)==len(classes))
	with open('nb_gauss.csv', 'w') as f:
		for i in range(len(samples)):
			f.write(','.join(samples[i].astype(str)) + ',' + str(classes[i]) + '\n')

def load_data_from_csv(file_path):
	samples = []
	classes = []
	with open(file_path, 'r') as f:
		for line in f.read().splitlines():
			values = line.split(',')
			sample_list = [values[0], values[1]]
			
			classes.append(int(values[2]))
			samples.append(sample_list)

	return np.array(samples), np.array(classes)


if __name__ == '__main__':
	means = np.array(MEANS)
	variances = np.array(VARS)

	# samples, classes = draw_gauss_samples(50, means, variances)

	# plot_2d_samples(samples, classes, means=means, variances=variances)

	# save_data_to_csv(samples, classes)

	model_means = np.array(MODEL_MEANS)
	model_vars = np.array(MODEL_VARS)

	samples, classes = load_data_from_csv('nb_gauss.csv')
	plot_model_vs_true(model_means, model_vars, means, variances, samples, classes)




