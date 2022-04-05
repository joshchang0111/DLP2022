import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")
	
	parser.add_argument("--result_path", type=str, default="../result")
	
	args = parser.parse_args()
	return args

def ReLU(x):
	"""ReLU"""
	return np.maximum(0, x)

def Leaky_ReLU(x, alpha=0.05):
	"""Leaky ReLU, alpha = 0.05"""
	return np.maximum(alpha * x, x)

def ELU(x, alpha=1):
	"""ELU, alpha = 1"""
	return np.where(x > 0, x, alpha * (np.exp(x) - 1))

if __name__ == "__main__":
	args = parse_args()

	x = np.linspace(-10, 10, 1000)
	#act_fcts = [ReLU, Leaky_ReLU, ELU]
	act_fcts = [ReLU, Leaky_ReLU, ELU]

	plt.figure()
	for act_fct in act_fcts:
		y = act_fct(x)
		plt.plot(x, y, label=act_fct.__doc__)

	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend()
	plt.savefig("{}/activations.png".format(args.result_path))
