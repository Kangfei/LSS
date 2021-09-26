import torch
import numpy as np
import random
from .active_util import _to_cuda, _to_dataloaders, _to_datasets, print_eval_res
from scipy.stats import  entropy, gmean
import datetime
import math

class ActiveLearner(object):
	def __init__(self, args):
		self.args = args
		self.budget = args.budget
		self.uncertainty = args.uncertainty
		self.active_iters = args.active_iters
		self.active_epochs = args.active_epochs
		self.biased_sample = args.biased_sample

	def train(self, model, criterion, criterion_cal,
			  train_datasets, val_datasets, optimizer, scheduler=None, active = False):
		if self.args.cuda:
			model.to(self.args.device)
		epochs = self.active_epochs if active else self.args.epochs

		train_loaders = _to_dataloaders(datasets= train_datasets)
		start = datetime.datetime.now()
		for loader_idx, dataloader in enumerate(train_loaders):
			model.train()
			print("Training the {}/{} Training set".format(loader_idx, len(train_loaders)))
			for epoch in range(epochs):
				epoch_loss, epoch_loss_cla = 0.0, 0.0
				for i, (decomp_x, decomp_edge_index, decomp_edge_attr, card, label) in \
						enumerate(dataloader):
					if self.args.cuda:
						decomp_x, decomp_edge_index, decomp_edge_attr = \
							_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
						card, label = card.cuda(), label.cuda()

					output, output_cla = model(decomp_x, decomp_edge_index, decomp_edge_attr)
					output = output.squeeze()
					# print("output_cla", output_cla)
					# print("label", label)
					loss = criterion(output, card)
					epoch_loss += loss.item()

					if self.args.multi_task and self.args.coeff > 0:
						loss_cla = criterion_cal(output_cla, label)
						loss += loss_cla * self.args.coeff
						epoch_loss_cla += loss_cla.item()
					loss.backward()

					if (i + 1) % self.args.batch_size == 0:
						optimizer.step()
						optimizer.zero_grad()

				if scheduler is not None and (epoch + 1) % self.args.decay_patience == 0:
					scheduler.step()
				print("{}-th QuerySet, {}-th Epoch: Reg. Loss={:.4f}, Cla. Loss={:.4f}"
					  .format(loader_idx, epoch, epoch_loss, epoch_loss_cla))

			# Evaluation the model
			all_eval_res = self.evaluate(model, criterion, val_datasets, print_res = True)
		end = datetime.datetime.now()
		elapse_time = (end - start).total_seconds()
		print("Training time: {:.4f}s".format(elapse_time))
		return model, elapse_time

	def evaluate(self, model, criterion, eval_datasets, print_res = False):
		if self.args.cuda:
			model.to(self.args.device)
		model.eval()
		all_eval_res = []
		eval_loaders = _to_dataloaders(datasets=eval_datasets)
		for loader_idx, dataloader in enumerate(eval_loaders):
			res = []
			loss, l1 = 0.0, 0.0
			start = datetime.datetime.now()
			for i, (decomp_x, decomp_edge_index, decomp_edge_attr, card, label) in \
					enumerate(dataloader):
				if self.args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
					card, label = card.cuda(), label.cuda()

				# print(decomp_x)
				output, output_cla = model(decomp_x, decomp_edge_index, decomp_edge_attr)
				output = output.squeeze()
				loss += criterion(card, output).item()
				l1 += torch.abs(card - output).item()

				res.append((card.item(), output.item()))
			end = datetime.datetime.now()
			elapse_time = (end - start).total_seconds()
			all_eval_res.append((res, loss, l1, elapse_time))

		if print_res:
			print_eval_res(all_eval_res)
		return all_eval_res

	def active_test(self, model, test_datasets, reject_set = None):
		assert self.args.multi_task, "Classification Task Disabled, Cannot Deploy Active Learning!"
		model.eval()
		test_uncertainties = []
		testset_dict = {}
		for dataset_idx, dataset in enumerate(test_datasets):
			for i in range(len(dataset)):
				# skip the test queries in the reject sets
				if reject_set is not None and (dataset_idx, i) in reject_set:
					continue
				decomp_x, decomp_edge_index, decomp_edge_attr, _, _ = dataset[i]
				if self.args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)

				# print(decomp_x)
				output, output_cla = model(decomp_x, decomp_edge_index, decomp_edge_attr)
				uncertainty = self.compute_uncertainty(output_cla, output)
				testset_dict[len(test_uncertainties)] = (dataset_idx, i)
				test_uncertainties.append(uncertainty)
		return self.select_active_sets(test_uncertainties, testset_dict, test_datasets)

	def select_active_sets(self, test_uncertainties, testset_dict, test_datasets):
		test_uncertainties = np.array(test_uncertainties)
		# select from testset
		test_uncertainties = test_uncertainties / np.sum(test_uncertainties)
		num_selected = self.budget if len(test_uncertainties) > self.budget else len(test_uncertainties)

		indices = np.random.choice(a=test_uncertainties.shape[0], size=num_selected, replace=False,
								   p=test_uncertainties) \
			if self.biased_sample else np.argsort(test_uncertainties)[- num_selected:]

		selected_set = [testset_dict[idx] for idx in indices]
		active_sets = []
		for dataset_idx, i in selected_set:
			decomp_x, decomp_edge_index, decomp_edge_attr, card = test_datasets[dataset_idx].queries[i]
			active_sets.append((decomp_x, decomp_edge_index, decomp_edge_attr, card))
		return active_sets, selected_set


	def compute_uncertainty(self, output_cal, output):
		assert self.uncertainty == "entropy" or self.uncertainty == "confident" or self.uncertainty == "margin" \
			   or self.uncertainty == "random" or self.uncertainty == "consist", \
			"Unsupported uncertainty criterion"
		output_cal, output = output_cal.squeeze(), output.squeeze()
		output_cal = torch.exp(output_cal) # transform to probability
		if self.args.cuda:
			output_cal = output_cal.cpu()
			output = output.cpu()
		output = output.item()
		output_cal = output_cal.detach().numpy()
		if self.uncertainty == "entropy":
			return entropy(output_cal)
		elif self.uncertainty == "confident":
			return 1.0 - np.max(output_cal)
		elif self.uncertainty == "margin":
			res = output_cal[np.argsort(output_cal)[-1]] - output_cal[np.argsort(output_cal)[-2]]
			return res
		elif self.uncertainty == "random":
			return random.random()
		elif self.uncertainty == "consist":
			reg_mag = math.ceil( math.log10( math.pow(2, output)))
			cla_mag = np.argmax(output_cal)
			return math.pow((reg_mag - cla_mag), 2)

	def merge_datasets(self, train_datasets, active_sets):
		active_train_datasets = []
		for dataset in train_datasets:
			active_train_datasets += dataset.queries
		active_train_datasets += active_sets
		return _to_datasets([active_train_datasets])

	def print_selected_set_info(self, selected_set):
		cnt_dict = {}
		for dataset_idx, i in selected_set:
			if dataset_idx not in cnt_dict.keys():
				cnt_dict[dataset_idx] = 0
			cnt_dict[dataset_idx] += 1
		print("Selected set info: # Selected Queries: {}".format(len(selected_set)))
		for key in sorted(cnt_dict.keys()):
			print("# Select Query in {}-th Test set: {}.".format(key, cnt_dict[key]))


	def active_train(self, model, criterion, criterion_cla, train_datasets, val_datasets, test_datasets, optimizer, scheduler=None, pretrain = True):

		reject_set = []
		if pretrain:
			model, _ = self.train(model, criterion, criterion_cla, train_datasets, val_datasets, optimizer, scheduler, active= False)
		active_train_datasets = train_datasets
		for iter in range(self.active_iters):

			active_sets, selected_set = self.active_test(model, test_datasets, reject_set)

			# merge the reject set
			reject_set += selected_set
			print("reject set size: {}".format(len(reject_set)))
			self.print_selected_set_info(reject_set)
			"""
			active_sets = _to_datasets(active_sets)
			active_train_datasets += active_sets
			active_train_datasets = ConcatDataset(active_train_datasets)
			"""
			active_train_datasets = self.merge_datasets(train_datasets, active_sets)
			print("The {}-th active Learning.".format(iter))
			model, _ = self.train(model, criterion, criterion_cla, active_train_datasets, val_datasets, optimizer, scheduler, active= True)


	def ensemble_evaluate(self, models, criterion, eval_datasets, print_res = False):
		"""
		get the final result of the ensemble models
		"""
		for model in models:
			if self.args.cuda:
				model.to(self.args.device)
			model.eval()
		all_eval_res = []
		eval_loaders = _to_dataloaders(datasets=eval_datasets)
		for loader_idx, dataloader in enumerate(eval_loaders):
			res = []
			loss, l1 = 0.0, 0.0
			start = datetime.datetime.now()
			for i, (decomp_x, decomp_edge_index, decomp_edge_attr, card, label) in \
					enumerate(dataloader):
				if self.args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
					card, label = card.cuda(), label.cuda()

				# print(decomp_x)
				# get the ensemble result
				outputs, losses, l1s = [], [], []
				for model in models:
					output, output_cla = model(decomp_x, decomp_edge_index, decomp_edge_attr)
					output = output.squeeze()
					outputs.append(output.item())
					losses.append(criterion(card, output).item())
					l1s.append(torch.abs(card - output).item())
				#print(outputs, losses, l1s)
				geo_output = gmean(outputs)
				loss += np.mean(losses)
				l1 += np.mean(l1s)

				res.append((card.item(), geo_output))
			end = datetime.datetime.now()
			elapse_time = (end - start).total_seconds()
			all_eval_res.append((res, loss, l1, elapse_time))

		if print_res:
			print_eval_res(all_eval_res)
		return all_eval_res



	def ensemble_active_test(self, models, test_datasets, reject_set = None):

		for model in models:
			model.eval()
		test_uncertainties = []
		testset_dict = {}
		for dataset_idx, dataset in enumerate(test_datasets):
			for i in range(len(dataset)):
				# skip the test queries in the reject sets
				if reject_set is not None and (dataset_idx, i) in reject_set:
					continue
				decomp_x, decomp_edge_index, decomp_edge_attr, _, _ = dataset[i]
				if self.args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)

				outputs = []
				for model in models:
					output, output_cla = model(decomp_x, decomp_edge_index, decomp_edge_attr)
					output = output.squeeze()
					if self.args.cuda:
						output = output.cpu()
					outputs.append(output.item())
				# uncertainty is the ensemble variance
				uncertainty = np.var(outputs)
				testset_dict[len(test_uncertainties)] = (dataset_idx, i)
				test_uncertainties.append(uncertainty)
		return self.select_active_sets(test_uncertainties, testset_dict, test_datasets)


	def ensemble_active_train(self, models, criterion, criterion_cla, train_datasets, val_datasets, test_datasets, optimizers, schedulers = None, pretrain = True):

		cur_models = []
		reject_set = []

		if pretrain: # pretrain all ensembled models
			for model, optimizer, scheduler in zip(models, optimizers, schedulers):
				model, _ = self.train(model, criterion, criterion_cla, train_datasets, val_datasets, optimizer, scheduler, active = False)
				cur_models.append(model)

		print("Ensemble Eval Result of {} Models:".format(len(cur_models)))
		self.ensemble_evaluate(cur_models, criterion, val_datasets, print_res=True)
		for iter in range(self.active_iters):
			active_sets, selected_set = self.ensemble_active_test(cur_models, test_datasets, reject_set)

			# merge the reject set
			reject_set += selected_set
			print("reject set size: {}".format(len(reject_set)))
			self.print_selected_set_info(reject_set)

			active_train_datasets = self.merge_datasets(train_datasets, active_sets)
			print("The {}-th active Learning.".format(iter))

			tmp_models = []
			for model, optimizer, scheduler in zip(cur_models, optimizers, schedulers):
				model, _ = self.train(model, criterion, criterion_cla, active_train_datasets, val_datasets, optimizer,
								  	scheduler, active=True)
				tmp_models.append(model)
			cur_models = tmp_models
			print("Ensemble Eval Result of {} Models:".format(len(cur_models)))
			self.ensemble_evaluate(cur_models, criterion, val_datasets, print_res=True)

