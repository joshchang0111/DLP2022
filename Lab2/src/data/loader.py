import numpy as np

def generate_dataset(data_type):
	print("Generating {} dataset.".format(data_type))

	if data_type.lower() == "linear":
		return generate_linear(n=100)
	elif data_type.lower() == "xor":
		return generate_XOR_easy()
	else:
		raise NotImplementedError("Function for generating \"{}\" not implemented!".format(data_type))

def generate_linear(n=100):
	"""Create linear inputs x, y."""
	pts = np.random.uniform(0, 1, (n, 2))
	inputs = []
	labels = []
	for pt in pts:
		inputs.append([pt[0], pt[1]])
		distance = (pt[0] - pt[1]) / 1.414
		if pt[0] > pt[1]:
			labels.append(0)
		else:
			labels.append(1)
	return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
	"""Create XOR inputs x, y."""
	inputs = []
	labels = []

	for i in range(11):
		inputs.append([0.1 * i, 0.1 * i])
		labels.append(0)

		if 0.1 * i == 0.5:
			continue

		inputs.append([0.1 * i, 1 - 0.1 * i])
		labels.append(1)

	return np.array(inputs), np.array(labels).reshape(21, 1)
