import os
import numpy as np
import torch


def make_dir(dir_str: str):
	if not os.path.exists(dir_str):
		os.makedirs(dir_str)

def save_true_card(card: str, card_save_path: str, run_time= None):
	with open(card_save_path, "w") as in_file:
		in_file.write(card + '\n')
		if run_time is not None:
			in_file.write(run_time + '\n')
		in_file.close()

def load_card(card_load_path: str):
	with open(card_load_path, "r") as in_file:
		card = in_file.readline().strip()
		card = int(card)
		in_file.close()
	return card


def load_card_time(card_load_path: str):
	with open(card_load_path, "r") as in_file:
		card = in_file.readline().strip()
		card = int(card)
		time = in_file.readline().strip()
		time = float(time)
		in_file.close()
	return card, time

def load_restore_dataset_hkust(graph_load_path: str, graph_save_path: str):
	nodes_list = []
	edges_list = []
	header = ""
	with open(graph_load_path, 'r' ) as in_file:
		for line in in_file:
			if line.strip().startswith("v"):
				tokens = line.strip().split()
				# v nodeID labelID degree
				id = int(tokens[1])
				label = tokens[2]
				nodes_list.append((id, label))
			elif line.strip().startswith("e"):
				tokens = line.strip().split()
				src, dst = int(tokens[1]), int(tokens[2])
				label = tokens[3] if len(tokens) >= 4 else "-1"
				edges_list.append((src, dst, label))
			else:
				header = line
		in_file.close()

	with open(graph_save_path, 'w') as out_file:
		out_file.write("{}".format(header))
		for v, label in nodes_list:
			out_file.write("v {} {}\n".format(v, label))
		for src, dst, label in edges_list:
			# no edge label for hkust dataset, transform to directed graph
			out_file.write("e {} {} {}\n".format(src, dst, -1))
			out_file.write("e {} {} {}\n".format(dst, src, -1))
		out_file.close()


def process_dataset_hkust_queries(true_card_dir: str, queries_load_path: str, queries_save_dir:str):
	make_dir(queries_save_dir)
	subsets_dir = os.listdir(true_card_dir)
	for subset_dir in subsets_dir:
		cards_dir = os.path.join(true_card_dir, subset_dir)
		if not os.path.isdir(cards_dir):
			continue
		pattern, size = str(subset_dir.split("_")[0]), str(subset_dir.split("_")[1])
		for card_dir in os.listdir(cards_dir):
			query_load_path = os.path.join(queries_load_path, os.path.splitext(card_dir)[0] + '.graph')
			queries_save_path = os.path.join(queries_save_dir, '_'.join([pattern, size]))
			make_dir(queries_save_path)
			query_save_path = os.path.join(queries_save_path, card_dir)
			load_restore_dataset_hkust(graph_load_path= query_load_path, graph_save_path= query_save_path)


def get_prediction_statistics(errors: list):
	print("<" * 80)
	print("Predict Result Profile:")
	print("Min/Max: {:.4f} / {:.4f}".format(np.min(errors), np.max(errors)))
	print("Mean: {:.4f}".format(np.mean(errors)))
	print("Median: {:.4f}".format(np.median(errors)))
	print("25%/75% Quantiles: {:.4f} / {:.4f}".format(np.quantile(errors, 0.25), np.quantile(errors, 0.75)))
	print(">" * 80)


def get_card_histogram(true_card_dir: str):
	import math
	cards_hist = {}
	subsets_dir = os.listdir(true_card_dir)
	num_zeros, num_queries = 0, 0
	for subset_dir in subsets_dir:
		cards_dir = os.path.join(true_card_dir, subset_dir)
		if not os.path.isdir(cards_dir):
			continue
		pattern, size = subset_dir.split("_")[0], int(subset_dir.split("_")[1])
		for card_dir in os.listdir(cards_dir):
			card_load_path = os.path.join(cards_dir, card_dir)
			card = load_card(card_load_path)
			#print(card)
			num_queries += 1
			if card ==  0:
				num_zeros += 1
			idx = math.ceil(math.log10(card + (10 ** -8)))
			if idx not in cards_hist.keys():
				cards_hist[idx] = 0
			cards_hist[idx] += 1

	print("# Total queries: {}".format(num_queries))
	print("# zero card: {}".format(num_zeros))
	for idx in sorted(cards_hist.keys()):
		print("# card in [10^{}, 10^{}]: {}".format(idx, idx+1, cards_hist[idx]))



def get_exact_runtime(true_card_dir: str, upper_card = 10 ** 10, lower_card = 0):
	subsets_dir = os.listdir(true_card_dir)
	num_queries = 0
	avg_runtime = {}
	size_cnt = {}
	for subset_dir in subsets_dir:
		cards_dir = os.path.join(true_card_dir, subset_dir)
		if not os.path.isdir(cards_dir):
			continue
		pattern, size = subset_dir.split("_")[0], int(subset_dir.split("_")[1])
		for card_dir in os.listdir(cards_dir):
			card_load_path = os.path.join(cards_dir, card_dir)
			card, runtime = load_card_time(card_load_path)
			if card >= upper_card or card < lower_card:
				continue
			if size not in avg_runtime.keys():
				avg_runtime[size] = 0.0
				size_cnt[size] = 0
			avg_runtime[size] += max(runtime, 0.000001)
			size_cnt[size] += 1
			num_queries += 1
	print("Exact running time of {} queries:".format(num_queries))
	for key, val in avg_runtime.items():
		print("Avg runtime for size {} query: {}".format(key, val/size_cnt[key]))




def compare_iso_homo_card(true_iso_dir: str, true_homo_dir:str):
	subsets_dir = os.listdir(true_iso_dir)
	results = []
	for subset_dir in subsets_dir:
		iso_cards_dir = os.path.join(true_iso_dir, subset_dir)
		homo_cards_dir = os.path.join(true_homo_dir, subset_dir)
		if not os.path.isdir(iso_cards_dir) or not os.path.isdir(iso_cards_dir):
			continue
		pattern, size = subset_dir.split("_")[0], int(subset_dir.split("_")[1])
		for card_dir in os.listdir(iso_cards_dir):
			iso_load_path = os.path.join(iso_cards_dir, card_dir)
			homo_load_path = os.path.join(homo_cards_dir, card_dir)
			if not os.path.exists(homo_load_path):
				continue
			iso_card = load_card(iso_load_path)
			homo_card = load_card(homo_load_path)
			results.append((card_dir, pattern, size, iso_card, homo_card))

	import csv
	with open("./iso_homo_card.csv", "a") as in_file:
		header = ['card_dir', 'pattern', 'size', 'iso_card', 'homo_card']
		writer = csv.writer(in_file, delimiter=',')
		if in_file.tell() == 0:
			writer.writerow(header)
		for (card_dir, pattern, size, iso_card, homo_card) in results:
			writer.writerow([card_dir, pattern, size, iso_card, homo_card])
		in_file.close()



def load_model(args, model, device, optimizer=None, scheduler=None):
	model_load_path = os.path.join(args.model_save_dir, args.dataset, args.model_file)
	assert os.path.isfile(model_load_path), "Cannot find model checkpoint!"
	model_snapshot = torch.load(model_load_path, map_location=device)
	model.load_state_dict(model_snapshot["model"])
	if optimizer is not None:
		optimizer.load_state_dict(model_snapshot["optimizer"])

	if scheduler is not None:
		scheduler.load_state_dict(model_snapshot["scheduler"])

def model_checkpoint(args, model, optimizer, scheduler=None, model_dir = None):
	if scheduler is not None:
		model_snapshot = {
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict()
		}
	else:
		model_snapshot = {
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict()
		}
	model_save_path = os.path.join(args.model_save_dir, args.dataset)
	make_dir(model_save_path)
	cumulative_flag = "cumu" if args.cumulative else "nocumu"

	if model_dir is not None:
		file_name = model_dir
	else:
		file_name = "model_{}_{}_{}.pth".format(args.model_type, args.epochs, cumulative_flag) if args.active_iters == 0 \
			else "model_{}_{}_{}_{}_{}_{}.pth".format(args.model_type, args.epochs, cumulative_flag, args.active_iters, args.uncertainty, args.budget)
	model_save_path = os.path.join(model_save_path, file_name)
	torch.save(model_snapshot, model_save_path)
	return file_name

def process_hin_data(path_dir, dataset:str):
	node_file = os.path.join(path_dir, 'node.dat')
	edge_files = [os.path.join(path_dir, 'link.dat')]
	outfile = os.path.join(path_dir, '{}.txt'.format(dataset))

	node_list = []
	edge_list = []
	with open(node_file, 'r') as node_input:
		for line in node_input:
			#print(line)
			tokens = line.strip().split('\t')
			node_id, node_type = tokens[0], tokens[2]
			node_list.append((node_id, node_type))
	node_input.close()
	for edge_file in edge_files:
		with open(edge_file, 'r') as edge_input:
			for line in edge_input:
				tokens = line.strip().split('\t')
				src, dst, edge_type = tokens[0], tokens[1], tokens[2]
				edge_list.append((src, dst, edge_type))
		edge_input.close()

	with open(outfile, 'w') as output_file:
		header = 't # {} {}\n'.format(len(node_list), len(edge_list))
		output_file.write(header)
		for (node_id, node_type) in node_list:
			output_file.write('v {} {}\n'.format(node_id, node_type))
		for (src, dst, edge_type) in edge_list:
			output_file.write('e {} {} {}\n'.format(src, dst, edge_type))
	output_file.close()



