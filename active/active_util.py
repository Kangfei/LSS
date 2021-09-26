import numpy as np
import random
import csv
import os
import math
from torch.utils.data import DataLoader
from Queryset import QueryDataset
from util import make_dir

def _to_cuda(l):
	"""
	put a list of tensor to gpu
	"""
	return [t.cuda() for t in l]


def _to_datasets(all_sets, num_classes = 10):
	datasets = [QueryDataset(queries=queries, num_classes=10)
				for queries in all_sets ] if isinstance(all_sets, list) \
		else [ QueryDataset(queries=all_sets) ]
	return datasets

def _to_dataloaders(datasets, batch_size = 1, shuffle = True):
	"""
	create a lists of torch dataloader from datasets
	"""

	dataloaders = [ DataLoader(dataset= dataset, batch_size = batch_size, shuffle= shuffle)
					for dataset in datasets ] if isinstance(datasets, list) \
		else [ DataLoader(dataset= datasets, batch_size = batch_size, shuffle= shuffle) ]
	return dataloaders

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def data_split_cv(all_sets, num_fold = 5, seed = 1):
	"""
	only support nocumulative learning currently
	"""
	random.seed(seed)
	all_fold_train_sets = []
	all_fold_val_sets = []
	for key in sorted(all_sets.keys()):
		random.shuffle(all_sets[key])

	for i in range(num_fold):
		train_sets = []
		val_sets = []
		for key in sorted(all_sets.keys()):
			# generate the key-th set for the i-th fold
			num_instances = int(len(all_sets[key]))
			num_fold_instances = num_instances / num_fold
			start = int(i * num_fold_instances)
			end = num_instances if i == num_fold - 1 else int(i * num_fold_instances + num_fold_instances)
			val_sets.append(all_sets[key][start: end])
			train_sets += (all_sets[key][: start] + all_sets[key][end:])
		all_fold_train_sets.append(train_sets)
		all_fold_val_sets.append(val_sets)
	return all_fold_train_sets, all_fold_val_sets

def get_prediction_statistics(errors: list):
	lower, upper = np.quantile(errors, 0.25), np.quantile(errors, 0.75)
	print("<" * 80)
	print("Predict Result Profile of {} Queries:".format(len(errors)))
	print("Min/Max: {:.4f} / {:.4f}".format(np.min(errors), np.max(errors)))
	print("Mean: {:.4f}".format(np.mean(errors)))
	print("Median: {:.4f}".format(np.median(errors)))
	print("25%/75% Quantiles: {:.4f} / {:.4f}".format(lower, upper))
	print(">" * 80)
	error_median = abs(upper - lower)
	return error_median



def get_prediction_statistics_exp(errors: list):
	print("<" * 80)
	print("Predict Result Profile of {} Queries:".format(len(errors)))
	print("Min/Max: {:.12f} / {:.12f}".format(math.pow(2, np.min(errors)), math.pow(2, np.max(errors))))
	print("Mean: {:.12f}".format(math.pow(2, np.mean(errors))))
	print("Median: {:.12f}".format(math.pow(2, np.median(errors))))
	print("25%/75% Quantiles: {:.12f} / {:.12f}".format(math.pow(2, np.quantile(errors, 0.25)), math.pow(2, np.quantile(errors, 0.75))))
	print(">" * 80)

def print_eval_res(all_eval_res, print_details= True):
	total_loss, total_l1 = 0.0, 0.0
	all_errors = []
	for i, (res, loss, l1, elapse_time) in enumerate(all_eval_res):
		print("Evaluation result of {}-th Eval set: Loss= {:.4f}, Avg. L1 Loss= {:.4f}, Avg. Pred. Time= {:.9f}(s)"
			  .format(i, loss, l1/len(res), elapse_time/len(res)))
		errors = [ (output - card) for card, output in res]
		get_prediction_statistics(errors)
		all_errors += errors
		total_loss += loss
		total_l1 += l1
		if print_details:
			for card, output in res:
				print("Card (log): {:.4f}, Pred (log) {:.4f}, Diff (log)= {:.4f}"
					  .format(card, output, output - card))
	print("Evaluation result of Eval dataset: Total Loss= {:.4f}, Total L1 Loss= {:.4f}".format(total_loss, total_l1))
	error_median = get_prediction_statistics(all_errors)
	return error_median

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
				#label = math.ceil(math.log10(math.pow(card, 2) + 10 ** (-8)))
				writer.writerow(["CardNet", size, math.pow(2, error), card])
		in_file.close()
