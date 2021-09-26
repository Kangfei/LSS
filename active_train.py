from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Queryset import Queryset
from QuerySampler import QueryDecompose

import torch.optim as optim
import torch
from active import ActiveLearner, _to_datasets, print_eval_res, data_split_cv, save_eval_res
import cardnet
import os
from util import model_checkpoint, load_model


def main(args):
	"""
	Entrance of train/test/active learning
	"""
	# input dir
	queryset_dir = args.queryset_dir
	true_card_dir = args.true_card_dir
	dataset = args.dataset
	data_dir = args.data_dir
	num_classes = args.max_classes

	# optimizer parameter
	lr = args.learning_rate
	weight_decay = args.weight_decay
	decay_factor = args.decay_factor


	QD = QueryDecompose(queryset_dir=queryset_dir, true_card_dir=true_card_dir, dataset=dataset, k=args.k)
	# decompose the query
	QD.decomose_queries()
	all_subsets = QD.all_subsets

	QS = Queryset(args= args, all_subsets=all_subsets)

	num_node_feat = QS.num_node_feat
	num_edge_feat = QS.num_edge_feat
	QS.print_queryset_info()

	train_sets, val_sets, test_sets, all_train_sets = QS.train_sets, QS.val_sets, QS.test_sets, QS.all_train_sets
	train_datasets = _to_datasets(train_sets, num_classes) if args.cumulative else _to_datasets(all_train_sets, num_classes)
	val_datasets, test_datasets, = _to_datasets(val_sets, num_classes), _to_datasets(test_sets, num_classes)

	model = cardnet.CardNet(args, num_node_feat= num_node_feat, num_edge_feat = num_edge_feat)
	print(model)
	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)

	active_learner = ActiveLearner(args)
	if args.mode == "train":
		print("start active learning ...")
		#load_model(args=args, model=model, device=args.device, optimizer=optimizer)
		active_learner.active_train(model=model, criterion=criterion, criterion_cla=criterion_cla,
								train_datasets=train_datasets, val_datasets=val_datasets, test_datasets=test_datasets,
								optimizer=optimizer, scheduler=scheduler, pretrain=True)
		model_checkpoint(args=args, model=model, optimizer=optimizer, scheduler=scheduler)

	elif args.mode == "pretrain":
		active_learner.active_train(model=model, criterion=criterion, criterion_cla=criterion_cla,
									train_datasets=train_datasets, val_datasets=val_datasets,
									test_datasets=test_datasets,
									optimizer=optimizer, scheduler=scheduler, pretrain=True)
		model_checkpoint(args=args, model=model, optimizer=optimizer, scheduler=scheduler)

	elif args.mode == "test":
		print("loading model ...")
		load_model(args = args, model=model, device=args.device, optimizer= optimizer)
		print("make prediction ...")
		active_learner.evaluate(model=model, criterion= criterion, eval_datasets=val_datasets, print_res=True)

def ensemble_learn(args):
	"""
	Entrance of Ensemble active learning
	"""
	# input dir
	queryset_dir = args.queryset_dir
	true_card_dir = args.true_card_dir
	dataset = args.dataset
	data_dir = args.data_dir

	# optimizer parameter
	lr = args.learning_rate
	weight_decay = args.weight_decay
	decay_factor = args.decay_factor


	QD = QueryDecompose(queryset_dir=queryset_dir, true_card_dir=true_card_dir, dataset=dataset, k=args.k)
	# decompose the query
	QD.decomose_queries()
	all_subsets = QD.all_subsets

	QS = Queryset(args= args, all_subsets=all_subsets)

	num_node_feat = QS.num_node_feat
	num_edge_feat = QS.num_edge_feat
	QS.print_queryset_info()

	train_sets, val_sets, test_sets, all_train_sets = QS.train_sets, QS.val_sets, QS.test_sets, QS.all_train_sets
	train_datasets = _to_datasets(train_sets) if args.cumulative else _to_datasets(all_train_sets)
	val_datasets, test_datasets, = _to_datasets(val_sets), _to_datasets(test_sets)

	models = []
	for _ in range(args.ensemble_num):
		models.append(cardnet.CardNet(args, num_node_feat= num_node_feat, num_edge_feat = num_edge_feat))

	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	optimizers = [ optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) for model in models]
	schedulers = [ optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor) for optimizer in optimizers]

	active_learner = ActiveLearner(args)
	active_learner.ensemble_active_train(models, criterion,criterion_cla,
										 train_datasets, val_datasets, test_datasets, optimizers, schedulers, pretrain=True)



def cross_validate(args):
	"""
	Entrance of cross validation, without active learning
	"""
	# input dir
	queryset_dir = args.queryset_dir
	true_card_dir = args.true_card_dir
	dataset = args.dataset
	num_classes = args.max_classes

	# optimizer parameter
	lr = args.learning_rate
	weight_decay = args.weight_decay
	decay_factor = args.decay_factor

	QD = QueryDecompose(queryset_dir=queryset_dir, true_card_dir=true_card_dir, dataset=dataset, k=args.k)
	# decompose the query
	QD.decomose_queries()
	all_subsets = QD.all_subsets

	QS = Queryset(args=args, all_subsets=all_subsets)
	num_node_feat = QS.num_node_feat
	num_edge_feat = QS.num_edge_feat
	QS.print_queryset_info()
	all_sizes = QS.all_sizes # {size -> (graphs, card)}
	all_fold_train_sets, all_fold_val_sets = data_split_cv(all_sizes, num_fold=args.num_fold)

	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	active_learner = ActiveLearner(args)
	all_fold_val_res = None
	i = 0
	total_elapse_time = 0.0
	for train_sets, val_sets in zip(all_fold_train_sets, all_fold_val_sets):
		i += 1
		print("start the {}/{} fold training ...".format(i, args.num_fold))
		train_datasets, val_datasets = _to_datasets([train_sets], num_classes), _to_datasets(val_sets, num_classes)
		model = cardnet.CardNet(args, num_node_feat=num_node_feat, num_edge_feat=num_edge_feat)
		print(model)
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)
		_, fold_elapse_time = active_learner.train(model=model, criterion=criterion, criterion_cal=criterion_cla,
									train_datasets=train_datasets, val_datasets=val_datasets,
									optimizer=optimizer, scheduler=scheduler, active=False)
		total_elapse_time += fold_elapse_time
		fold_eval_res = active_learner.evaluate(model=model, criterion=criterion, eval_datasets=val_datasets)
		# merge the result of the evaluation result of each fold
		if all_fold_val_res is None:
			all_fold_val_res = fold_eval_res
		else:
			tmp_all_fold_val_res = []
			for all_res, fold_res in zip(all_fold_val_res, fold_eval_res):
				tmp_res = all_res[0] + fold_res[0]
				tmp_loss = all_res[1] + fold_res[1]
				tmp_l1 = all_res[2] + fold_res[2]
				tmp_elapse_time = all_res[3] + fold_res[3]
				tmp_all_fold_val_res.append((tmp_res, tmp_loss, tmp_l1, tmp_elapse_time))
			all_fold_val_res = tmp_all_fold_val_res
	print("the average training time: {:.4f}(s)".format(total_elapse_time / args.num_fold))
	print("the total evaluation result:")
	error_median = print_eval_res(all_fold_val_res, print_details=False)
	print("error_median={}".format(0 - error_median))
	save_eval_res(args, sorted(all_sizes.keys()), all_fold_val_res, args.save_res_dir)


if __name__ == "__main__":
	parser = ArgumentParser("LSS", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	# Model Settings (ONLY FOR CardNet MODEL)
	parser.add_argument("--num_layers", default=3, type=int,
						help="number of convolutional layers")
	parser.add_argument("--model_type", default="GIN", type=str,
						help="GNN layer type") # GIN, GINE, GAT, NN, GCN, SAGE, NNGIN, NNGINConcat
	parser.add_argument("--embed_type", default="freq", type=str,
						help="the node feature encoding type") # freq, n2v, prone, n2v_concat, prone_concat, nrp are tested
	parser.add_argument("--edge_embed_type", default="freq", type=str,
						help="the edge feature encoding type")
	parser.add_argument("--num_g_hid", default=128, type=int,
						help="hidden dim for transforming nodes for intermediate GNN layer")
	parser.add_argument("--num_e_hid", default=32, type=int,
						help="hidden dim for transforming edges for intermediate GNN layer")
	parser.add_argument("--out_g_ch", default=128, type=int,
						help="number of output dimension of the final GNN layer")
	parser.add_argument("--num_expert", default=64, type=int,
						help="hyper-parameter for the attention layer")
	parser.add_argument("--num_att_hid", default=64, type=int,
						help="hyper-parameter for the attention layer")
	parser.add_argument("--num_mlp_hid", default=128, type=int,
						help="number of hidden units of MLP")
	parser.add_argument('--pool_type', type=str, default="att",  # att, mean, sum, max
						help='shards pooling layer type')
	parser.add_argument('--dropout', type=float, default=0.2,
						help='Dropout rate (1 - keep probability).')
	# Training settings
	parser.add_argument("--cumulative", default=False, type=bool,
						help='Whether or not to enable cumulative learning')
	parser.add_argument("--num_fold", default=5, type=int,
						help="number of fold for cross validation")
	parser.add_argument("--epochs", default=80, type=int)
	parser.add_argument("--batch_size", default= 2, type=int)
	parser.add_argument("--learning_rate", default= 1e-4, type=float)
	parser.add_argument('--weight_decay', type=float, default=5e-4,
						help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--decay_factor', type=float, default=0.1,
						help='decay rate of (gamma).')
	parser.add_argument('--decay_patience', type=int, default=50,
						help='num of epochs for one lr decay.')
	parser.add_argument('--weight_exp', type=float, default=1.0,
						help='loss weight exp factor.')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='Disables CUDA training.')
	parser.add_argument('--num_workers', type = int, default= 16,
						help='number of workers for Dataset.')
	# Classification task settings
	parser.add_argument("--multi_task", default=True, type=bool,
						help="enable/disable card classification task.")
	parser.add_argument("--max_classes", default=10, type=int,
						help="number classes for the card classification task.")
	parser.add_argument('--coeff', type=float, default=0.5,
						help='coefficient for the classification loss.')
	# Active Learner settings
	parser.add_argument("--uncertainty", default="consist", type=str,
						help="The uncertainty type") # entropy, margin, confident, consist, random are tested
	parser.add_argument("--biased_sample", default=True, type=bool,
						help="Enable Biased sampling for test set selection")
	parser.add_argument('--active_iters', type=int, default=2,
						help='Num of iterators of active learning.')
	parser.add_argument('--budget', type=int, default=50,
						help='Selected Queries budget Per Iteration.')
	parser.add_argument('--active_epochs', type=int, default=50,
						help='Training Epochs for per iteration active learner.')
	parser.add_argument('--ensemble_num', type=int, default=5,
						help='number of ensemble models for active learning.')
	# Input and Output directory
	parser.add_argument("--dataset", type=str, default="aids")  # aids, wordnet, yeast, hprd, youtube, eu2005 are tested
	parser.add_argument("--full_data_dir", type=str, default="./data/")
	parser.add_argument("--save_res_dir", type=str, default="./result/")
	parser.add_argument("--model_file", type=str, default="aids_homo.pth")
	parser.add_argument("--model_save_dir", type=str, default="./models")

	# Other parameters
	parser.add_argument("--matching", default="homo", type=str,
						help="The subgraph matching mode")
	parser.add_argument('--k', type=int, default=30,
						help='decompose hop number.')
	parser.add_argument("--verbose", default=True, type=bool)
	parser.add_argument("--mode", default="cross_val", type=str,
						help="The running mode") # train (train & test) or test (only test) or pretrain or ensemble or cross_val
	args = parser.parse_args()

	# set the hardware parameter
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.device = torch.device('cuda' if args.cuda else 'cpu')
	# set the input dir
	args.queryset_iso_dir = os.path.join(args.full_data_dir, "queryset")
	args.queryset_homo_dir = os.path.join(args.full_data_dir, "queryset_homo")
	args.true_iso_dir = os.path.join(args.full_data_dir, "true_cardinality")
	args.true_homo_dir = os.path.join(args.full_data_dir, "true_homo")
	args.data_dir = os.path.join(args.full_data_dir, "dataset")
	args.prone_feat_dir = os.path.join(args.full_data_dir, "prone")
	args.n2v_feat_dir = os.path.join(args.full_data_dir, "n2v")
	args.nrp_feat_dir = os.path.join(args.full_data_dir, "nrp")

	args.embed_feat_dir = args.n2v_feat_dir if args.embed_type == "n2v" or args.embed_type == "n2v_concat" else \
		args.prone_feat_dir
	if args.embed_type == "nrp":
		args.embed_feat_dir = args.nrp_feat_dir
	args.active_iters = 0 if args.mode == "pretrain" else args.active_iters
	args.queryset_dir = args.queryset_homo_dir if args.matching == "homo" else  args.queryset_iso_dir
	args.true_card_dir = args.true_homo_dir if args.matching == "homo" else args.true_iso_dir


	if args.verbose:
		print(args)
	if args.mode == "cross_val":
		cross_validate(args)
	elif args.mode == "ensemble":
		ensemble_learn(args)
	else: # train/test/pre-train
		main(args)
