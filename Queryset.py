import os
import pickle
import torch
import networkx as nx
import numpy as np
import random
import statistics
import math

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class Queryset(object):
	def __init__(self, args, all_subsets):
		"""
		all_subsets: {(size, patten) -> [(graphs, true_card]} // all queries subset
		"""
		self.args = args
		self.data_dir = args.data_dir
		self.dataset = args.dataset
		self.num_queries = 0

		# load the data graph and its statistical information
		self.data_graph = DataGraph(data_dir = self.data_dir, dataset = self.dataset)
		self.node_label_card, self.edge_label_card = self.data_graph.node_label_card, self.data_graph.edge_label_card

		self.num_nodes = self.data_graph.num_nodes
		self.num_edges = self.data_graph.num_edges
		self.node_label_fre = 0
		self.edge_label_fre = 0


		self.label_dict = self.load_label_mapping() if self.dataset == 'aids' or self.dataset == 'aids_gen' else \
			{key: key for key in self.node_label_card.keys()}  # label_id -> embed_id
		embed_feat_path = os.path.join(args.embed_feat_dir, "{}.emb.npy".format(args.dataset))
		embed_feat = np.load(embed_feat_path)

		#assert embed_feat.shape[0] == len(self.node_label_card) + 1, "prone embedding size error!"
		self.embed_dim = embed_feat.shape[1]
		self.embed_feat = torch.from_numpy(embed_feat)
		if self.args.embed_type == "freq":
			self.num_node_feat = len(self.node_label_card)
		elif self.args.embed_type == "n2v" or self.args.embed_type == "prone" or self.args.embed_type == "nrp":
			self.num_node_feat = self.embed_dim
		else:
			self.num_node_feat = self.embed_dim + len(self.node_label_card)

		if self.args.edge_embed_type == "freq":
			self.edge_embed_feat = None
			self.edge_embed_dim = 0
			self.num_edge_feat = len(self.edge_label_card)
		else:
			edge_embed_feat_path = os.path.join(args.embed_feat_dir, "{}_edge.emb.npy".format(args.dataset))
			edge_embed_feat = np.load(edge_embed_feat_path)
			self.edge_embed_dim = edge_embed_feat.shape[1]
			self.edge_embed_feat = torch.from_numpy(edge_embed_feat)
			self.num_edge_feat = self.edge_embed_dim


		# transform the decomposed query to torch tensor
		self.all_subsets = self.transform_query_to_tensors(all_subsets)

		self.all_queries = []
		# split by query patterns and sizes
		self.all_sizes = {} # { size -> [(graphs, true_card]} // all queries subset indexed by query size
		self.all_patterns = {} # { pattern -> [(graphs, true_card)]} // all queries subset index by query pattern

		for (pattern, size), graphs_card_pairs in self.all_subsets.items():
			self.all_queries += graphs_card_pairs
			if pattern not in self.all_patterns.keys():
				self.all_patterns[pattern] = []
			self.all_patterns[pattern] += graphs_card_pairs
			if size not in self.all_sizes.keys():
				self.all_sizes[size] = []
			self.all_sizes[size] += graphs_card_pairs


		# split to train, val, test query set
		self.num_train_queries , self.num_val_queries, self.num_test_queries = 0, 0, 0
		train_sets, val_sets, test_sets, all_train_sets = self.data_split(self.all_sizes, train_ratio= 0.8, val_ratio=0.1)
		#train_sets, val_sets, test_sets, all_train_sets = self.uneven_data_split(self.all_sizes, train_ratio=0.5,
		#																  val_ratio=0.4)

		self.train_loaders = self.to_dataloader(all_sets= train_sets)
		self.val_loaders = self.to_dataloader(all_sets= val_sets)
		self.test_loaders = self.to_dataloader(all_sets= test_sets)
		self.all_train_loaders = self.to_dataloader(all_sets = all_train_sets)
		self.train_sets, self.val_sets, self.test_sets, self.all_train_sets = train_sets, val_sets, test_sets, all_train_sets


	def data_split(self, all_sets, train_ratio = 0.6, val_ratio = 0.2, seed = 1):
		"""
		all_sets: self.all_sets or self.all_patterns
		test_ratio = 1 - train_ratio - val_ratio
		"""
		assert train_ratio + val_ratio <= 1.0, "Error data split ratio!"
		random.seed(seed)
		train_sets, val_sets, test_sets = [], [], []
		all_train_sets = [ [] ]
		for key in sorted(all_sets.keys()):
			num_instances = len(all_sets[key])
			random.shuffle(all_sets[key])
			train_sets.append(all_sets[key][ : int(num_instances * train_ratio)])
			# merge to all_train_sets
			all_train_sets[-1] = all_train_sets[-1] + train_sets[-1]
			val_sets.append(all_sets[key][int(num_instances * train_ratio): int(num_instances * (train_ratio + val_ratio))])
			test_sets.append(all_sets[key][int(num_instances * (train_ratio + val_ratio)): ])
			self.num_train_queries += len(train_sets[-1])
			self.num_val_queries += len(val_sets[-1])
			self.num_test_queries += len(test_sets[-1])
		return train_sets, val_sets, test_sets, all_train_sets

	def uneven_data_split(self, all_sets, train_ratio=0.6, val_ratio=0.2, large_ratio = 0.6, small_ratio = 0.4, seed=1):
		"""
		split a training query set with uneven large/small query graphs
		large_ratio: specified large to small query ratio in the training set
		all_sets: self.all_sizes
		test_ratio = 1 - train_ratio - val_ratio
		"""
		assert train_ratio + val_ratio <= 1.0, "Error train/val/test data split ratio!"
		assert large_ratio + small_ratio == 1.0, "Error large/small data split ratio!"
		print("large/small ratio: {}/{}".format(large_ratio, small_ratio))
		random.seed(seed)
		tmp_train_sets = {}
		train_sets, val_sets, test_sets = [], [], []
		# Split the train/val/test dataset
		for key in sorted(all_sets.keys()):
			num_instances = len(all_sets[key]) # queries in the current subset
			random.shuffle(all_sets[key])
			tmp_train_sets[key] = all_sets[key][ : int(num_instances * train_ratio)]
			val_sets.append(all_sets[key][int(num_instances * train_ratio): int(num_instances * (train_ratio + val_ratio))])
			test_sets.append(all_sets[key][int(num_instances * (train_ratio + val_ratio)): ])
			self.num_val_queries += len(val_sets[-1])
			self.num_test_queries += len(test_sets[-1])
		median_size = statistics.median(list(all_sets.keys()))
		small_sizes = [size for size in list(all_sets.keys()) if size <= median_size]
		large_sizes = [size for size in list(all_sets.keys()) if size > median_size]
		all_train_sets = [[]]
		# sample an uneven training dataset
		for key in sorted(all_sets.keys()):
			num_instances = len(tmp_train_sets[key])
			if key in small_sizes:
				train_sets.append(random.sample(tmp_train_sets[key], k=int(num_instances * small_ratio)))
			elif key in large_sizes:
				train_sets.append(random.sample(tmp_train_sets[key], k=int(num_instances * large_ratio)))
			all_train_sets[-1] = all_train_sets[-1] + train_sets[-1]
			self.num_train_queries += len(train_sets[-1])
		return train_sets, val_sets, test_sets, all_train_sets


	def to_dataloader(self, all_sets, batch_size = 1, shuffle = True):
		datasets = [ QueryDataset(queries= queries)
					 for queries in all_sets]
		dataloaders = [ DataLoader(dataset= dataset, batch_size = batch_size, shuffle= shuffle)
						for dataset in datasets]
		return dataloaders


	def transform_query_to_tensors(self, all_subsets):
		tmp_subsets = {}
		for (pattern, size), graphs_card_pairs in all_subsets.items():
			tmp_subsets[(pattern, size)] = []
			for (graphs, card) in graphs_card_pairs:
				decomp_x, decomp_edge_index, decomp_edge_attr, _ = \
					self._get_decomposed_graph_data(graphs)
				tmp_subsets[(pattern, size)].append((decomp_x, decomp_edge_index, decomp_edge_attr, card))
				self.num_queries += 1

		return tmp_subsets


	def _get_decomposed_graph_data(self, graphs, card = None):
		decomp_x = []
		decomp_edge_index = []
		decomp_edge_attr = []
		for graph in graphs:
			if self.args.embed_type == "freq":
				node_attr = self._get_nodes_attr_freq(graph)
			elif self.args.embed_type == "n2v" or self.args.embed_type == "prone" or self.args.embed_type == "nrp":
				node_attr = self._get_nodes_attr_embed(graph)
			else:
				node_attr_freq, node_attr_embed = self._get_nodes_attr_freq(graph), self._get_nodes_attr_embed(graph)
				node_attr = torch.cat([node_attr_freq, node_attr_embed], dim=1)

			if self.args.edge_embed_type == "freq":
				edge_index, edge_attr = self._get_edges_index_freq(graph)
			else:
				edge_index, edge_attr = self._get_edges_index_embed(graph)
			decomp_x.append(node_attr)
			decomp_edge_index.append(edge_index)
			decomp_edge_attr.append(edge_attr)
		return decomp_x, decomp_edge_index, decomp_edge_attr, card

	def _get_nodes_attr(self, graph):
		node_attr = torch.zeros(size=(graph.number_of_nodes(), self.num_node_feat), dtype= torch.float)
		for v in graph.nodes():
			if len(graph.nodes[v]["labels"]) == 0:
				continue
			for label in graph.nodes[v]["labels"]:
				node_attr[v] += self.embed_feat[self.label_dict[label]]
				self.node_label_fre += 1
		return node_attr

	def _get_nodes_attr_freq(self, graph):
		node_attr = torch.ones(size=(graph.number_of_nodes(), len(self.node_label_card)), dtype=torch.float)
		for v in graph.nodes():
			for label in graph.nodes[v]["labels"]:
				node_attr[v][self.label_dict[label]] = self.node_label_card[label]
				self.node_label_fre += 1
		return node_attr

	def _get_nodes_attr_embed(self, graph):
		node_attr = torch.zeros(size=(graph.number_of_nodes(), self.embed_dim), dtype=torch.float)
		for v in graph.nodes():
			if len(graph.nodes[v]["labels"]) == 0:
				continue
			for label in graph.nodes[v]["labels"]:
				node_attr[v] += self.embed_feat[self.label_dict[label]]
				self.node_label_fre += 1
		return node_attr


	def _get_edges_index_freq(self, graph):
		edge_index = torch.ones(size= (2, graph.number_of_edges()), dtype = torch.long)
		edge_attr = torch.zeros(size= (graph.number_of_edges(), len(self.edge_label_card)), dtype=torch.float)
		cnt = 0
		for e in graph.edges():
			edge_index[0][cnt], edge_index[1][cnt] = e[0], e[1]
			for label in graph.edges[e]["labels"]:
				edge_attr[cnt][label] = self.edge_label_card[label]
				self.edge_label_fre += 1
			cnt += 1
		return edge_index, edge_attr

	def _get_edges_index_embed(self, graph):
		edge_index = torch.ones(size= (2, graph.number_of_edges()), dtype = torch.long)
		edge_attr = torch.zeros(size= (graph.number_of_edges(), self.edge_embed_dim), dtype=torch.float)
		cnt = 0
		for e in graph.edges():
			edge_index[0][cnt], edge_index[1][cnt] = e[0], e[1]
			for label in graph.edges[e]["labels"]:
				edge_attr[cnt] += self.edge_embed_feat[label]
				self.edge_label_fre += 1
			cnt += 1
		return edge_index, edge_attr

	def load_label_mapping(self):
		map_load_path = os.path.join(self.args.embed_feat_dir, "{}_mapping.txt".format(self.dataset))
		assert os.path.exists(map_load_path), "The label mapping file is not exists!"
		label_dict = {} # label_id -> embed_id
		cnt = 0
		with open(map_load_path, "r") as in_file:
			for line in in_file:
				label_id = int(line.strip())
				label_dict[label_id] = cnt
				cnt += 1
			in_file.close()
		return label_dict

	def print_queryset_info(self):
		print("<" * 80)
		print("Query Set Profile:")
		print("# Total Queries: {}".format(self.num_queries))
		print("# Train Queries: {}".format(self.num_train_queries))
		print("# Val Queries: {}".format(self.num_val_queries))
		print("# Test Queries: {}".format(self.num_test_queries))
		print("# Node Feat: {}".format(self.num_node_feat))
		print("# Edge Feat: {}".format(self.num_edge_feat))
		print("# Node label fre: {}".format(self.node_label_fre))
		print("# Edge label fre: {}".format(self.edge_label_fre))
		print(">" * 80)




class DataGraph(object):
	def __init__(self, data_dir, dataset):
		self.data_dir = data_dir
		self.data_set = dataset
		self.max_node_label_num, self.min_node_label_num = 0, float('inf')
		self.max_edge_label_num, self.min_edge_label_num = 0, float('inf')
		self.data_load_path = os.path.join(data_dir, dataset, "{}.txt".format(dataset))
		self.graph, self.node_label_card, self.edge_label_card = self.load_graph()
		self.num_nodes = self.graph.number_of_nodes()
		self.num_edges = self.graph.number_of_edges()
		self.max_deg = 0
		for u in self.graph.nodes():
			self.max_deg = max(self.max_deg, self.graph.degree(u))
		self.print_data_graph_info()


	def load_graph(self):
		file = open(self.data_load_path)
		nodes_list = []
		edges_list = []
		node_label_card = {}
		edge_label_card = {}

		for line in file:
			if line.strip().startswith("v"):
				tokens = line.strip().split()
				# v nodeID label1 label2 ... (may have multiple labels)
				id = int(tokens[1])
				labels = [int(token) for token in tokens[2:]]
				#labels = [] if -1 in tmp_labels else tmp_labels
				for label in labels:
					if label not in node_label_card.keys():
						node_label_card[label] = 1.0
					else:
						node_label_card[label] += 1.0
				nodes_list.append((id, {"labels": labels}))
				self.max_node_label_num = max(self.max_node_label_num, len(labels))
				self.min_node_label_num = min(self.min_node_label_num, len(labels))

			if line.strip().startswith("e"):
				tokens = line.strip().split()
				src, dst = int(tokens[1]), int(tokens[2])
				labels = [int(token) for token in tokens[3:]]
				#labels = [] if -1 in tmp_labels else tmp_labels
				for label in labels:
					if label not in edge_label_card.keys():
						edge_label_card[label] = 1.0
					else:
						edge_label_card[label] += 1.0
				edges_list.append((src, dst, {"labels": labels}))
				self.max_edge_label_num = max(self.max_edge_label_num, len(labels))
				self.min_edge_label_num = min(self.min_edge_label_num, len(labels))

		graph = nx.Graph()
		graph.add_nodes_from(nodes_list)
		graph.add_edges_from(edges_list)

		file.close()

		# nomarlized the node/edge label card
		for key, val in node_label_card.items():
			node_label_card[key] = val / graph.number_of_nodes()

		from scipy.stats import entropy
		print("data graph label entropy: {}".format(entropy(list(node_label_card.values()))))

		for key, val in edge_label_card.items():
			edge_label_card[key] = val / graph.number_of_edges()
		return graph, node_label_card, edge_label_card

	def print_data_graph_info(self):
		print("<" * 80)
		print("Data Graph {} Profile:".format(self.data_set))
		print("# Nodes: {}".format(self.graph.number_of_nodes()))
		print("# Edges: {}".format(self.graph.number_of_edges()))
		print("# of Node Labels: {}".format(len(self.node_label_card)))
		print("# of Edge Labels: {}".format(len(self.edge_label_card)))
		print("Max/Min Node Labels: {}/{}".format(self.max_node_label_num, self.min_node_label_num))
		print("Max/Min Edge Labels: {}/{}".format(self.max_edge_label_num, self.min_edge_label_num))
		print("Max Degree: {}".format(self.max_deg))
		print(">" * 80)


class QueryDataset(Dataset):
	def __init__(self, queries, num_classes = 10):
		"""
		parameter:
		all_queries =[(decomp_x, decomp_edge_attr, decomp_edge_attr, card)]
		num_classes: number of classes for the classification task
		"""
		self.queries = queries
		self.label_base = 10
		self.num_classes = num_classes

	def __len__(self):
		return len(self.queries)

	def __getitem__(self, index):
		"""
		decomp_x, decomp_edge_attr, decomp_edge_attr: list[Tensor]
		"""
		decomp_x, decomp_edge_index, decomp_edge_attr, card = self.queries[index]
		idx = math.ceil(math.log(card, self.label_base))
		idx = self.num_classes - 1 if idx >= self.num_classes else idx
		card = torch.tensor(math.log(card, 2), dtype=torch.float)
		label = torch.tensor(idx, dtype=torch.long)

		return decomp_x, decomp_edge_index, decomp_edge_attr, card, label



