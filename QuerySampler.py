import os

import networkx as nx
import random
import queue
import pickle
from util import make_dir, save_true_card
import tqdm
from multiprocessing import Pool
import functools

def load_graph(fname):
	file = open(fname)
	nodes_list = []
	edges_list = []
	label_cardinality = {}

	for line in file:
		if line.strip().startswith("v"):
			tokens = line.strip().split()
			# v nodeID labelID degree
			id = int(tokens[1])
			labels = tokens[2]
			if labels not in label_cardinality.keys():
				label_cardinality[labels] = 1
			else:
				label_cardinality[labels] += 1
			nodes_list.append((id, {"labels": labels}))
		if line.strip().startswith("e"):
			tokens = line.strip().split()
			src, dst = int(tokens[1]), int(tokens[2])
			labels = [] #tokens[3:]
			edges_list.append((src, dst, {"labels" : labels}))

	graph = nx.Graph()
	graph.add_nodes_from(nodes_list)
	graph.add_edges_from(edges_list)

	print('number of nodes: {}'.format(graph.number_of_nodes()))
	print('number of edges: {}'.format(graph.number_of_edges()))
	file.close()
	return graph

def save_graph(graph, graph_save_path):
	with open(graph_save_path, 'w') as file:
		file.write("t {} {} \n".format(graph.number_of_nodes(), graph.number_of_edges()))
		for v in graph.nodes():
			file.write("v {} {} {} \n".format(v, graph.nodes[v]['labels'], graph.degree[v]))
		for (u, v) in graph.edges():
			file.write("e {} {} \n".format(u, v))


class QuerySampler(object):
	def __init__(self, graph):
		self.graph = graph


	def sample_chain(self, node_num):
		nodes_list = []
		edges_list = []

		src = random.randint(0, self.graph.number_of_nodes())
		Q = queue.Queue()
		Q.put(src)
		while not Q.empty():
			cur = Q.get()
			if len(nodes_list) > 0:
				edges_list.append((nodes_list[-1], cur))
			nodes_list.append(cur)
			if len(nodes_list) == node_num:
				break

			candidates = set(list(self.graph.neighbors(cur))).difference(set(nodes_list))
			if len(candidates) == 0:
				continue
			next = random.choice(list(candidates))
			Q.put(next)

		if len(nodes_list) < node_num:
			return None
		sample = self.node_reorder(nodes_list, edges_list)

		return sample

	def sample_star(self, node_num):
		nodes_list = []
		edges_list = []
		while True:
			src = random.randint(0, self.graph.number_of_nodes())
			if self.graph.degree[src] >= node_num - 1:
				break
		nodes_list.append(src)
		nexts = random.sample(list(self.graph.neighbors(src)), k= node_num - 1)
		for v in nexts:
			nodes_list.append(v)
			edges_list.append((src, v))
		sample = self.node_reorder(nodes_list, edges_list)
		return  sample

	def sample_tree(self, node_num):
		nodes_list = []
		edges_list = []
		parent = {}

		src = random.randint(0, self.graph.number_of_nodes())
		Q = queue.Queue()
		Q.put(src)
		while not Q.empty():
			cur = Q.get()
			if len(nodes_list) > 0:
				edges_list.append((parent[cur], cur))
			nodes_list.append(cur)
			if len(nodes_list) == node_num:
				break

			candidates = set(list(self.graph.neighbors(cur))).difference(set(nodes_list))
			if len(candidates) == 0:
				continue
			nexts = random.sample(list(self.graph.neighbors(src)), k = random.randint(1, min(len(candidates), node_num - len(nodes_list))))
			for v in nexts:
				Q.put(v)
				parent[v] = cur

		sample = self.node_reorder(nodes_list, edges_list)
		return sample

	def sample_cycle(self, node_num):
		nodes_list = [(0, {"labels" : random.randint(0, 7)})]
		edges_list = []

		for v in range(1, node_num):
			nodes_list.append((v, {"labels" : random.randint(0, 7)}))
			edges_list.append((v - 1, v))
		edges_list.append((node_num - 1, 0))
		sample = nx.Graph()
		sample.add_nodes_from(nodes_list)
		sample.add_edges_from(edges_list)
		return sample

	def sample_clique(self, node_num):
		nodes_list = []
		edges_list = []
		for v in range(0, node_num):
			nodes_list.append((v, {"labels": random.randint(0, 7)}))
			for u in range(0, v):
				edges_list.append((u, v))
		sample = nx.Graph()
		sample.add_nodes_from(nodes_list)
		sample.add_edges_from(edges_list)
		return sample

	def node_reorder(self, nodes_list, edges_list):
		idx_dict = {}
		node_cnt = 0
		for v in nodes_list:
			idx_dict[v] = node_cnt
			node_cnt += 1
		nodes_list = [(idx_dict[v], {"labels": self.graph.nodes[v]["labels"]})
					  for v in nodes_list]
		edges_list = [(idx_dict[u], idx_dict[v], {"labels": self.graph.edges[u, v]["labels"]})
					  for (u, v) in edges_list]
		sample = nx.Graph()
		sample.add_nodes_from(nodes_list)
		sample.add_edges_from(edges_list)
		return sample

class QueryDecompose(object):
	def __init__(self, queryset_dir: str, true_card_dir: str, dataset: str, k = 3):
		"""
		load the query graphs, true counts and perform query decomposition
		"""
		self.queryset = queryset_dir
		self.dataset = dataset
		self.queryset_load_path = os.path.join(queryset_dir, dataset)
		self.true_card_dir = true_card_dir
		self.true_card_load_path = os.path.join(true_card_dir, dataset)
		self.k = k
		self.num_queries = 0
		self.all_subsets = {} # {(size, patten) -> [(decomp_graphs, true_card]}
		# preserve the undecomposed queries
		self.all_queries = {} # {(size, patten) -> [(graph, card)]}
		self.lower_card = 10 ** 0
		self.upper_card = 10 ** 20



	def decomose_queries(self):
		avg_label_den = 0.0
		distinct_card = {}
		subsets_dir = os.listdir(self.queryset_load_path)
		for subset_dir in subsets_dir:
			queries_dir = os.path.join(self.queryset_load_path, subset_dir)
			if not os.path.isdir(queries_dir):
				continue
			pattern, size = subset_dir.split("_")[0], int(subset_dir.split("_")[1])
			self.all_subsets[(pattern, size)] = []
			self.all_queries[(pattern, size)] = []
			for query_dir in os.listdir(queries_dir):
				query_load_path = os.path.join(self.queryset_load_path, subset_dir, query_dir)
				card_load_path = os.path.join(self.true_card_load_path, subset_dir, query_dir)
				if not os.path.isfile(query_load_path) or os.path.splitext(query_load_path)[1] == ".pickle":
					continue
				# load, decompose the query
				query, label_den = self.load_query(query_load_path)
				avg_label_den += label_den
				graphs = self.decompose(query)
				true_card = self.load_card(card_load_path)
				if true_card >=  self.upper_card or true_card < self.lower_card:
					continue
				true_card = true_card + 1 if true_card == 0 else true_card
				self.all_subsets[(pattern, size)].append((graphs, true_card))
				self.all_queries[(pattern, size)].append((query, true_card))
				self.num_queries += 1
				# save the decomposed query
				#query_save_path = os.path.splitext(query_load_path)[0] + ".pickle"
				#self.save_decomposed_query(graphs, true_card, query_save_path)
				#print("save decomposed query: {}".format(query_save_path))
		print("average label density: {}".format(avg_label_den/self.num_queries))


	def decompose(self, query):
		graphs = []
		for src in query.nodes():
			G = self.k_hop_spanning_tree(query, src)
			graphs.append(G)
		return graphs

	def k_hop_spanning_tree(self, query, src):
		nodes_list = [src]
		edges_list = []
		Q = queue.Queue()
		Q.put(src)
		depth = 0
		while not Q.empty():
			s = Q.qsize()
			for _ in range(s):
				cur = Q.get()
				for next in query.neighbors(cur):
					if next in nodes_list:
						continue
					Q.put(next)
					nodes_list.append(next)
					edges_list.append((cur, next))
			depth += 1
			if depth >= self.k:
				break

		G = self.node_reorder(query, nodes_list, edges_list)
		return G

	def k_hop_induced_subgraph(self, query, src):
		nodes_list = [src]
		Q = queue.Queue()
		Q.put(src)
		depth = 0
		while not Q.empty():
			s = Q.qsize()
			for _ in range(s):
				cur = Q.get()
				for next in query.neighbors(cur):
					if next in nodes_list:
						continue
					Q.put(next)
					nodes_list.append(next)
			depth += 1
			if depth >= self.k:
				break
		edges_list = query.subgraph(nodes_list).edges()
		G = self.node_reorder(query, nodes_list, edges_list)
		return G

	def node_reorder(self, query, nodes_list, edges_list):
		idx_dict = {}
		node_cnt = 0
		for v in nodes_list:
			idx_dict[v] = node_cnt
			node_cnt += 1
		nodes_list = [(idx_dict[v], {"labels": query.nodes[v]["labels"]})
					  for v in nodes_list]
		edges_list = [(idx_dict[u], idx_dict[v], {"labels": query.edges[u, v]["labels"]})
					  for (u, v) in edges_list]
		sample = nx.Graph()
		sample.add_nodes_from(nodes_list)
		sample.add_edges_from(edges_list)
		return sample

	def load_query(self, query_load_path):
		file = open(query_load_path)
		nodes_list = []
		edges_list = []
		label_cnt = 0

		for line in file:
			if line.strip().startswith("v"):
				tokens = line.strip().split()
				# v nodeID labelID
				id = int(tokens[1])
				tmp_labels = [int(tokens[2])] # (only one label in the query node)
				#tmp_labels = [int(token) for token in tokens[2 : ]]
				labels = [] if -1 in tmp_labels else tmp_labels
				label_cnt += len(labels)
				nodes_list.append((id, {"labels": labels}))

			if line.strip().startswith("e"):
				# e srcID dstID labelID1 labelID2....
				tokens = line.strip().split()
				src, dst = int(tokens[1]), int(tokens[2])
				tmp_labels = [int(tokens[3])]
				#tmp_labels = [int(token) for token in tokens[3 : ]]
				labels = [] if -1 in tmp_labels else tmp_labels
				edges_list.append((src, dst, {"labels": labels}))

		query = nx.Graph()
		query.add_nodes_from(nodes_list)
		query.add_edges_from(edges_list)

		#print('number of nodes: {}'.format(graph.number_of_nodes()))
		#print('number of edges: {}'.format(graph.number_of_edges()))
		file.close()
		label_den = float(label_cnt) / query.number_of_nodes()
		return query, label_den

	def load_card(self, card_load_path):
		with open(card_load_path, "r") as in_file:
			card = in_file.readline().strip()
			card = int(card)
			in_file.close()
		return card

	def save_decomposed_query(self, graphs, card, save_path):
		with open(save_path, "wb") as out_file:
			obj = {"graphs": graphs, "card": card}
			pickle.dump(obj = obj, file=out_file, protocol=3)
			out_file.close()



def get_true_cardinality(card_estimator_path, query_load_path, graph_load_path, timeout_sec = 7200):
	import subprocess
	est_path = os.path.join(card_estimator_path, "SubgraphMatching.out")
	cmd = "timeout %d %s -d %s -q %s -filter GQL -order GQL -engine LFTJ -num MAX"\
		  %(timeout_sec, est_path, graph_load_path, query_load_path)
	print(cmd)
	popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
	popen.wait()
	card, run_time = None, None
	for line in iter(popen.stdout.readline, b''):
		line = line.decode("utf-8")
		if line.startswith("#Embeddings:"):
			card = line.partition("#Embeddings:")[-1].strip()
		elif line.startswith("Enumerate time (seconds):"):
			run_time = line.partition("Enumerate time (seconds):")[-1].strip()
	return card, run_time


def get_batch_true_card(card_estimator_path, queries_load_path, graph_load_path, card_save_dir):
	queries_dir = os.listdir(queries_load_path)
	for query_dir in queries_dir:
		query_load_path = os.path.join(queries_load_path, query_dir)
		pattern, size, = str(query_dir.split("_")[1]), str(query_dir.split("_")[2])

		card, run_time = get_true_cardinality(card_estimator_path, query_load_path, graph_load_path)
		if card is not None:
			card_save_path = os.path.join(card_save_dir, '_'.join([pattern, size]))
			make_dir(card_save_path)
			card_save_path = os.path.join(card_save_path, os.path.splitext(query_dir)[0] + '.txt')
			save_true_card(card, card_save_path, run_time)
			print("save card {} in {}".format(card, card_save_path))


def get_save_true_card(card_estimator_path, graph_load_path, card_save_dir, query_dir):
	query_load_path = os.path.join(queries_load_path, query_dir)
	pattern, size, = str(query_dir.split("_")[1]), str(query_dir.split("_")[2])

	card, run_time = get_true_cardinality(card_estimator_path, query_load_path, graph_load_path)
	if card is not None:
		card_save_path = os.path.join(card_save_dir, '_'.join([pattern, size]))
		make_dir(card_save_path)
		card_save_path = os.path.join(card_save_path, os.path.splitext(query_dir)[0] + '.txt')
		save_true_card(card, card_save_path, run_time)
		print("save card {} in {}".format(card, card_save_path))


def process_batch_true_card(card_estimator_path, queries_load_path, graph_load_path, card_save_dir, num_workers = 10):
	"""
	parallel version of get_batch_true_card
	"""
	pro = functools.partial(get_save_true_card, card_estimator_path,
											 graph_load_path, card_save_dir)
	queries_dir = os.listdir(queries_load_path)
	p = Pool(num_workers)
	p.map(pro, queries_dir)


