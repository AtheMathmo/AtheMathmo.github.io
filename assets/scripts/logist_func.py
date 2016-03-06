import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

if __name__ == '__main__':
	x = np.arange(-8,8,0.1)
	y = sigmoid(x)

	plt.figure(figsize=(14,8))
	plt.plot(x,y, lw = 2)
	plt.axhline(y=0.5, color='black')

	plt.title(r"The Logistic Function, $h(x) = \frac{1}{1 + e^{-x}}$", fontsize=28, y=1.04)
	plt.xlabel(r"$x$", fontsize=22)
	plt.ylabel(r"$h(x)$", fontsize=22)

	plt.savefig("../logist_func.png")
	plt.show()