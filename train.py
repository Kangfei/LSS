from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Queryset import Queryset
from QuerySampler import QueryDecompose
from util import get_prediction_statistics
import torch.optim as optim
import torch
from cardnet import CardNet


def _to_cuda(l):
	"""
	put a list of tensor to gpu
	"""
	return [t.cuda() for t in l]


def evaluate(args, model, criterion, eval_loaders):
	if args.cuda:
		model.to(args.device)
	model.eval()
	all_eval_res = []
	for loader_idx, dataloader in enumerate(eval_loaders):
		res = []
		loss, l1 = 0.0, 0.0
		for i, (decomp_x, decomp_edge_index, decomp_edge_attr, card, label) in \
				enumerate(dataloader):
			if args.cuda:
				decomp_x, decomp_edge_index, decomp_edge_attr = \
					_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
				card, label = card.cuda(), label.cuda()

			#print(decomp_x)
			output, output_cla = model(decomp_x, decomp_edge_index, decomp_edge_attr)
			output = output.squeeze()
			loss += criterion(card, output).item()
			l1 += torch.abs(card - output).item()

			res.append((card.item(), output.item()))
		all_eval_res.append((res, loss, l1))
	return all_eval_res

def print_eval_res(all_eval_res):
	for i, (res, loss, l1) in enumerate(all_eval_res):
		print("Evaluation result of {}-th Eval set: Loss= {:.4f}, Avg. L1 Loss= {:.4f}"
			  .format(i, loss, l1/len(res)))
		errors = [ (output - card) for card, output in res]
		get_prediction_statistics(errors)
		for card, output in res:
			print("Card (log): {:.4f}, Pred (log) {:.4f}, Diff (log)= {:.4f}"
				  .format(card, output, output - card))


def train(args, model, criterion, train_loaders, val_loaders, optimizer,  scheduler = None):
	if args.cuda:
		model.to(args.device)
	nll_loss = torch.nn.NLLLoss()
	for loader_idx, dataloader in enumerate(train_loaders):
		model.train()
		print("Training the {}/{} Training set".format(loader_idx, len(train_loaders)))
		for epoch in range(args.epochs):
			epoch_loss, epoch_loss_cla = 0.0, 0.0
			for i , (decomp_x, decomp_edge_index, decomp_edge_attr, card, label) in \
				enumerate(dataloader):
				if args.cuda:
					decomp_x, decomp_edge_index, decomp_edge_attr = \
						_to_cuda(decomp_x), _to_cuda(decomp_edge_index), _to_cuda(decomp_edge_attr)
					card, label = card.cuda(), label.cuda()

				output, output_cla = model(decomp_x, decomp_edge_index, decomp_edge_attr)
				output= output.squeeze()
				#print("output_cla", output_cla)
				#print("label", label)
				loss = criterion(output, card)
				epoch_loss += loss.item()

				if args.multi_task and args.coeff > 0:
					loss_cla = nll_loss(output_cla, label)
					loss += loss_cla * args.coeff
					epoch_loss_cla += loss_cla.item()
				loss.backward()

				if (i + 1) % args.batch_size == 0:
					optimizer.step()
					optimizer.zero_grad()

			if scheduler is not None and (epoch + 1) % args.decay_patience == 0:
				scheduler.step()
			print("{}-th QuerySet, {}-th Epoch: Reg. Loss={:.4f}, Cla. Loss={:.4f}"
				  .format(loader_idx, epoch, epoch_loss, epoch_loss_cla))

		# Evaluation the model
		all_eval_res = evaluate(args, model, criterion, val_loaders)
		print_eval_res(all_eval_res)

	return model

def main(args):
	# input dir
	queryset_dir = args.queryset_dir
	true_card_dir = args.true_card_dir
	dataset = args.dataset
	data_dir = args.data_dir

	# model parameter
	num_g_hid = args.num_g_hid
	out_g_ch = args.out_g_ch
	num_expert = args.num_expert
	num_att_hid = args.num_att_hid
	v_hid = args.v_hid
	num_mlp_hid = args.num_mlp_hid
	dropout = args.dropout

	# optimizer parameter
	lr = args.learning_rate
	weight_decay = args.weight_decay
	decay_factor = args.decay_factor

	# hardware parameter
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.device = torch.device('cuda' if args.cuda else 'cpu')

	QD = QueryDecompose(queryset_dir=queryset_dir, true_card_dir=true_card_dir, dataset=dataset, k=3)
	# decompose the query
	QD.decomose_queries()
	all_subsets = QD.all_subsets

	QS = Queryset(args= args, all_subsets=all_subsets)

	num_node_feat = QS.num_node_feat
	num_edge_feat = QS.num_edge_feat
	QS.print_queryset_info()

	train_loaders, val_loaders, test_loaders, all_train_loaders = \
		QS.train_loaders, QS.val_loaders, QS.test_loaders, QS.all_train_loaders

	"""
	model = MPNN(num_node_feat= num_node_feat, num_edge_feat = num_edge_feat,
				 num_g_hid= num_g_hid, out_g_ch= out_g_ch,
				 num_expert=num_expert, num_att_hid= num_att_hid, v_hid= v_hid,
				 num_mlp_hid=num_mlp_hid, dropout= dropout)
	"""
	model = CardNet(args, num_node_feat= num_node_feat, num_edge_feat = num_edge_feat)
	print(model)
	criterion = torch.nn.MSELoss()
	criterion_cla = torch.nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)


	print("start cumulative training...")
	model = train(args, model = model, criterion= criterion, train_loaders=train_loaders,
				  val_loaders= val_loaders, optimizer= optimizer, scheduler= scheduler)
	"""
	print("start regular learning ..")
	model = train(args, model=model, criterion=criterion, train_loaders=all_train_loaders,
				  val_loaders=val_loaders, optimizer=optimizer, scheduler=scheduler)
	print("Training End.")
	"""



if __name__ == "__main__":
	parser = ArgumentParser("NeuroGraphCard", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
	# Model Settings (ONLY FOR MPNN MODEL)
	parser.add_argument("--num_layers", default=3, type=int,
						help="number of convolutional layers")
	parser.add_argument("--model_type", default="GIN", type=str, # GIN, GINE, GAT, NN
						help="GNN type")
	parser.add_argument("--num_g_hid", default=64, type=int,
						help="hidden dim for transforming nodes")
	parser.add_argument("--num_e_hid", default=4, type=int,
						help="hidden dim for transforming edges")
	parser.add_argument("--out_g_ch", default=64, type=int,
						help="number of output dimension")
	parser.add_argument("--num_expert", default=32, type=int,
						help="number of output units of MPNN")
	parser.add_argument("--num_att_hid", default=64, type=int,
						help="number of hidden units of edge network")
	parser.add_argument("--v_hid", default=64, type=int,
						help="number of hidden units of MLP")
	parser.add_argument("--num_mlp_hid", default=32, type=int,
						help="number of hidden units of MLP")
	parser.add_argument('--dropout', type=float, default=0.5,
						help='Dropout rate (1 - keep probability).')
	parser.add_argument("--batch_norm", default=False, type=bool)
	# Card Classification talks
	parser.add_argument("--multi_task", default=False, type=bool,
						help= "enable/disable card classification task.")
	parser.add_argument("--max_classes", default=10, type=int,
						help="number classes for the card classification task.")
	parser.add_argument('--coeff', type=float, default=0.5,
						help='coefficient for the classification loss.')
	# Training settings
	parser.add_argument("--num_fold", default=5, type=int,
						help="number of fold for cross validation")
	parser.add_argument("--epochs", default=50, type=int)
	parser.add_argument("--batch_size", default= 1, type=int)
	parser.add_argument("--learning_rate", default= 5e-4, type=float)
	parser.add_argument('--weight_decay', type=float, default=5e-4,
						help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--decay_factor', type=float, default=0.9,
						help='decay rate of (gamma).')
	parser.add_argument('--decay_patience', type=int, default=10,
						help='num of epoches for one lr decay.')
	parser.add_argument('--weight_exp', type=float, default=1.0,
						help='loss weight exp factor.')
	parser.add_argument('--no-cuda', action='store_true', default=True,
						help='Disables CUDA training.')
	parser.add_argument('--num_workers', type = int, default= 16,
						help='number of workers for Dataset.')
	# Active Learner settings
	parser.add_argument("--uncertainty", default="entropy", type=str,  # entroy, margin, confident
						help="The uncertainty type")

	# Input dir
	parser.add_argument("--dataset", type=str, default="yeast")
	parser.add_argument("--queryset_dir", type=str, default="/home/kfzhao/GraphCard/data/queryset/")
	parser.add_argument("--true_card_dir", type=str, default="/home/kfzhao/GraphCard/data/true_cardinality")
	parser.add_argument("--data_dir", type=str, default="/home/kfzhao/GraphCard/data/dataset/")
	parser.add_argument("--verbose", default=True, type=bool)

	# decompose parameter
	parser.add_argument('--k', type=int, default=3,
						help='decompose hop number.')

	args = parser.parse_args()
	if args.verbose:
		print(args)
	main(args)