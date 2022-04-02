import ipdb
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
	parser = argparse.ArgumentParser(description="Rumor Detection")

	parser.add_argument("--plot_which", type=str, default="lrs")
	parser.add_argument("--lrs", type=str, default="lr0.1,lr0.05,lr0.01,lr0.0025")
	parser.add_argument("--tuples", type=str, default="ori,tuple1,tuple2,ori+tuple1")

	parser.add_argument("--result_path", type=str, default="../result")
	parser.add_argument("--figure_path", type=str, default="../figure")

	args = parser.parse_args()

	return args

def plot(episodes, data, title, label=""):
	## Avoid too many ticks
	episodes = np.asarray(episodes, float)
	data = np.asarray(data, float)

	plt.plot(episodes, data, label=label)
	plt.xlabel("Episode", fontsize=15)
	plt.title(title, fontsize=15)

if __name__ == "__main__":
	args = parse_args()

	statistics = [
		"mean", 
		"max", 
		"win_rate_2048"
	]

	result_dict = {}
	if args.plot_which == "lrs":
		exps = args.lrs.split(",")
	elif args.plot_which == "tuples":
		exps = args.tuples.split(",")

	save_unit = 1000
	for exp in exps:
		result_file = "{}/{}/statistic.txt".format(args.result_path, exp)
		episodes, means, maxs, win_rate_2048s = [], [], [], []
	
		print("Reading statistic file from {}".format(result_file))
		for idx, line in enumerate(open(result_file).readlines()):
			line = line.strip().rstrip()
			episode, mean, max_, win_rate_2048 = line.split("\t")
			
			if episode == "episode":
				continue
		
			if idx % save_unit == 0:
				episodes.append(episode)
				means.append(mean)
				maxs.append(max_)
				win_rate_2048s.append(win_rate_2048)

		result_dict[exp] = {
			"episode": episodes, 
			"mean": means, 
			"max": maxs, 
			"win_rate_2048": win_rate_2048s
		}

	## Plot and save figures
	max_plot_episode = len(result_dict[exps[0]]["episode"])
	if args.plot_which == "tuples":
		max_plot_episode = int(max_plot_episode * 0.6)

	for statistic in statistics:
		plt.figure() ## Plot different lrs on same figure
		for exp in exps:
			if args.plot_which == "lrs":
				label = "{} {}".format(exp[:2], exp[2:])
			elif args.plot_which == "tuples":
				label = exp.replace("tuple", "Group ").replace("ori", "Group 0").replace("+", " + ")

			plot(
				result_dict[exp]["episode"][:max_plot_episode], 
				result_dict[exp][statistic][:max_plot_episode], 
				title=statistic.capitalize().replace("_", " "), 
				label=label
			)
		
		if len(exps) == 1:
			file_name = exps[0]
		else:
			file_name = "all_{}".format(args.plot_which)
			plt.legend()
		fig_file = "{}/{}/{}.png".format(args.figure_path, statistic, file_name)
		plt.savefig(fig_file)
		print("Save figure to {}".format(fig_file))
