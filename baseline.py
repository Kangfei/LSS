import os
import math
import csv
from active import get_prediction_statistics_exp, get_prediction_statistics

def load_restore_baseline(res_load_dir: str, dataset: str, tar_method: str):
	res_load_dir = os.path.join(res_load_dir, dataset)
	all_method_files = os.listdir(res_load_dir)
	all_est_res = {} # {(pattern, size, query_name)-> (est_card, run_time)}
	for method_file in all_method_files:

		method = str(method_file.strip().split('_')[1])
		if not method == tar_method:
			continue
		if os.path.splitext(method_file)[1] != ".txt":
			continue
		res_load_path = os.path.join(res_load_dir, method_file)
		with open(res_load_path, "r") as in_file:
			while True:
				line1 = in_file.readline()
				line2 = in_file.readline()
				if not line1 or not line2:
					break
				tokens = line1.strip().split()
				file_name, est_card, run_time = str(tokens[0]), float(tokens[1]), float(tokens[2])
				file_name_tokens = file_name.strip().split("/")
				pattern, size = file_name_tokens[-2].split("_")[0], int(file_name_tokens[-2].split("_")[1])
				if est_card < 0:
					continue
				est_card = 0.5 if est_card < 1 else est_card
				query_name = file_name_tokens[-1].strip()
				all_est_res[(pattern, size, query_name)] = (est_card, run_time)
	return all_est_res

def load_restore_estimate(res_load_dir:str, dataset: str, embed_type:str):

	method_file_name = "{}_GIN_{}_80_cv.csv".format(dataset, embed_type)
	file_path = os.path.join(res_load_dir, dataset, method_file_name)
	all_range_card = {}
	all_size_card = {}
	with open(file_path, "r") as in_file:
		in_file.readline() # skip the csv header
		for line in in_file:
			tokens = line.strip().split(",")
			#print(tokens)
			size, error, true_card = int(tokens[1]), float(tokens[2]), float(tokens[3])
			range = math.ceil(math.log10(math.pow(2.0, true_card)))
			if range == 0:
				range += 1
			if range not in all_range_card.keys():
				all_range_card[range] = []
			all_range_card[range].append(error)

			if size not in all_size_card.keys():
				all_size_card[size] = []
			all_size_card[size].append(error)

	for range in sorted(all_range_card.keys()):
		print("Error of true card in [10^{},  10^{}]".format(range -1, range))
		get_prediction_statistics(all_range_card[range])

	for size in sorted(all_size_card.keys()):
		print("Error of queries with size {}".format(size))
		get_prediction_statistics(all_size_card[size])



def load_restore_baseline1800(res_load_dir: str, dataset: str, tar_method: str):
	res_load_dir = os.path.join(res_load_dir, dataset)
	all_method_files = os.listdir(res_load_dir)
	all_est_res = {} # {(pattern, size, query_name)-> (est_card, run_time)}
	for method_file in all_method_files:

		method = str(method_file.strip().split('_')[1])
		if not method == tar_method:
			continue
		if os.path.splitext(method_file)[1] != ".txt":
			continue
		res_load_path = os.path.join(res_load_dir, method_file)
		with open(res_load_path, "r") as in_file:
			while True:
				line1 = in_file.readline()
				line2 = in_file.readline()
				if not line1 or not line2:
					break
				tokens = line1.strip().split()
				file_name, est_card, run_time = str(tokens[0]), float(tokens[1]), float(tokens[2])
				file_name_tokens = file_name.strip().split("/")
				pattern, size = file_name_tokens[-1].split("_")[1], int(file_name_tokens[-1].split("_")[2])
				if est_card < 0:
					continue
				est_card = 1.0 if est_card <= 1 else est_card
				query_name = file_name_tokens[-1].strip()
				all_est_res[(pattern, size, query_name)] = (est_card, run_time)
	return all_est_res

"""
def save_eval_res(args, sizes, all_eval_res, save_res_dir):
	make_dir(save_res_dir)
	save_res_dir = os.path.join(save_res_dir, args.dataset)
	make_dir(save_res_dir)
	save_res_path = os.path.join(save_res_dir, "{}_{}_{}_{}_cv.csv".format(args.dataset, args.model_type, args.embed_type, args.epochs))
	header = ['method', 'size', 'error', 'true_card']
	with open(save_res_path, 'w') as in_file:
		writer = csv.writer(in_file, delimiter=',')
		if in_file.tell() == 0:
			writer.writerow(header)
		for size, eval_res in zip(sizes, all_eval_res):
			res = eval_res[0]
			for card, output in res:
				error = output - card
				label = math.ceil(math.log10(math.pow(card, 2) + 10 ** (-8)))
				writer.writerow(["CardNet", size, math.pow(2, error), label])
		in_file.close()
"""

class EvalBaseline(object):
	def __init__(self, true_card_dir, dataset):

		self.dataset = dataset
		self.true_card_dir = true_card_dir
		self.true_card_load_path = os.path.join(true_card_dir, dataset)
		self.num_queries = 0

	def eval_cards(self, all_est_res, upper_card = 10 ** 7, lower_card = 0):
		# load true_card and eval with true_card
		all_sizes_card = {}
		all_sizes_run_time = {}
		max_est = 0
		fail_cnt ={}
		subsets_dir = os.listdir(self.true_card_load_path)
		for subset_dir in subsets_dir:
			cards_dir = os.path.join(self.true_card_load_path, subset_dir)
			if not os.path.isdir(cards_dir):
				continue
			pattern, size = subset_dir.split("_")[0], int(subset_dir.split("_")[1])
			for card_dir in os.listdir(cards_dir):
				card_load_path = os.path.join(self.true_card_load_path, subset_dir, card_dir)
				if not os.path.isfile(card_load_path):
					continue
				#print(card_load_path)
				true_card = self.load_card(card_load_path)
				# filter large card query and query does not estimated result
				if true_card >=  upper_card or true_card < lower_card:
					continue
				if (pattern, size, card_dir) not in all_est_res.keys():
					continue

				est_card, run_time = all_est_res[(pattern, size, card_dir)]

				if est_card == 1.0:
					if size not in fail_cnt.keys():
						fail_cnt[size] = 0
					fail_cnt[size] += 1
				if size not in all_sizes_card.keys():
					all_sizes_card[size] = []
					all_sizes_run_time[size] = 0.0
				#all_sizes_card[size].append((true_card, est_card))
				max_est = max(max_est, est_card)
				all_sizes_card[size].append((math.log2(true_card), math.log2(est_card)))
				all_sizes_run_time[size] += run_time
				self.num_queries += 1
		for key in sorted(fail_cnt.keys()):
			print("# failure for size {} queries: {}".format(key, fail_cnt[key]))
		print("max est card :{}".format(max_est))
		return all_sizes_card, all_sizes_run_time

	def eval_range_cards(self, all_est_res, upper_card = 10 ** 7, lower_card = 0):
		# load true_card and eval with true_card
		all_range_card = {}
		all_range_run_time = {}
		true_card_dict = {}
		max_est = 0
		fail_cnt = {}
		subsets_dir = os.listdir(self.true_card_load_path)
		for subset_dir in subsets_dir:
			cards_dir = os.path.join(self.true_card_load_path, subset_dir)
			if not os.path.isdir(cards_dir):
				continue
			pattern, size = subset_dir.split("_")[0], int(subset_dir.split("_")[1])
			for card_dir in os.listdir(cards_dir):
				card_load_path = os.path.join(self.true_card_load_path, subset_dir, card_dir)
				if not os.path.isfile(card_load_path):
					continue
				# print(card_load_path)
				true_card = self.load_card(card_load_path)
				#print(true_card)
				if true_card not in true_card_dict.keys():
					true_card_dict[true_card] = 0
				true_card_dict[true_card] += 1
				# filter large card query and query does not estimated result
				if true_card >= upper_card or true_card < lower_card:
					continue
				if (pattern, size, card_dir) not in all_est_res.keys():
					continue

				est_card, run_time = all_est_res[(pattern, size, card_dir)]
				range = math.ceil(math.log10(true_card))
				if range == 0:
					range += 1
				if est_card == 0.5:
					if size not in fail_cnt.keys():
						fail_cnt[size] = 0
					fail_cnt[size] += 1
				if range not in all_range_card.keys():
					all_range_card[range] = []
					all_range_run_time[range] = 0.0
				# all_sizes_card[size].append((true_card, est_card))
				max_est = max(max_est, est_card)
				all_range_card[range].append((math.log2(true_card), math.log2(est_card)))
				all_range_run_time[range] += run_time
				self.num_queries += 1
		print("distinct true card: {}".format(len(true_card_dict.keys())))
		for key in sorted(fail_cnt.keys()):
			print("# failure for size {} queries: {}".format(key, fail_cnt[key]))
		print("max est card :{}".format(max_est))
		return all_range_card, all_range_run_time

	def save_est_res(self, all_sizes_card):
		pass

	def load_card(self, card_load_path):
		with open(card_load_path, "r") as in_file:
			card = in_file.readline().strip()
			card = int(card)
			in_file.close()
		return card




if __name__ == "__main__":
	res_load_dir = "/home/kfzhao/GraphCard/data/raw_res/"
	model_res_load_dir = "/home/kfzhao/GraphCard/result/"
	true_card_dir = "/home/kfzhao/GraphCard/data/true_homo"
	dataset = "yago"
	tar_method = "sumrdf"
	embed_type = "prone_concat"
	all_methods = ["wj", "cset", "cs", "jsub", "impr", "bsk", "sumrdf"]
	all_est_res = load_restore_baseline(res_load_dir, dataset, tar_method)
	print("all est res", len(all_est_res))
	for k, val in all_est_res.items():
		print(k, val)
	evaluation =EvalBaseline(true_card_dir= true_card_dir, dataset= dataset)

	all_sizes_card, all_sizes_run_time = evaluation.eval_cards(all_est_res, upper_card=10 ** 20, lower_card=10 ** 0)
	#all_sizes_card, all_sizes_run_time = evaluation.eval_range_cards(all_est_res, upper_card=10 ** 20, lower_card=10 ** 0)
	print("total evaluated queries: {}".format(evaluation.num_queries))
	i = 0
	for size in sorted(all_sizes_card.keys()):
		print("Avg. Runtime of {} on {}-th query set: {:.4f}"
			  .format(tar_method, i, all_sizes_run_time[size] / len(all_sizes_card[size])))
		errors = [(est - card) for card, est in all_sizes_card[size]]
		#errors = [ est/card for card, est in all_sizes_card[size]]
		#get_prediction_statistics(errors)
		get_prediction_statistics_exp(errors)
		i += 1
	#load_restore_estimate(res_load_dir=model_res_load_dir, dataset=dataset, embed_type=embed_type)