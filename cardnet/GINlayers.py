import torch
from typing import Union, Tuple, Callable
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform, zeros


class NNGINConv(MessagePassing):
	"""
	Add the node embedding with edge embedding
	"""
	def __init__(self, edge_nn: Callable, node_nn: Callable,
				 eps: float = 0.,train_eps: bool = False, aggr: str = 'add', **kwargs):
		super(NNGINConv, self).__init__(aggr=aggr, **kwargs)
		self.edge_nn = edge_nn
		self.node_nn = node_nn
		self.aggr = aggr
		self.initial_eps = eps
		if train_eps:
			self.eps = torch.nn.Parameter(torch.Tensor([eps]))
		else:
			self.register_buffer('eps', torch.Tensor([eps]))


	def reset_parameters(self):
		reset(self.node_nn)
		reset(self.edge_nn)
		self.eps.data.fill_(self.initial_eps)

	def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
		if isinstance(x, Tensor):
			x: OptPairTensor = (x, x)

		# propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
		out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
		x_r = x[1]
		if x_r is not None:
			out += (1 + self.eps) * x_r

		out = self.node_nn(out)
		#print(out.shape)
		return out

	def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
		edge_attr = self.edge_nn(edge_attr)
		return F.relu(x_j + edge_attr)


	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.edge_nn,
                                   self.node_nn)

class NNGINConcatConv(MessagePassing):
	"""
	Concatenate the node embedding with edge embedding
	no self loop
	"""
	def __init__(self, edge_nn: Callable, node_nn: Callable,
				 eps: float = 0.,train_eps: bool = False, aggr: str = 'add', **kwargs):
		super(NNGINConcatConv, self).__init__(aggr=aggr, **kwargs)
		self.edge_nn = edge_nn
		self.node_nn = node_nn
		self.aggr = aggr
		self.initial_eps = eps
		if train_eps:
			self.eps = torch.nn.Parameter(torch.Tensor([eps]))
		else:
			self.register_buffer('eps', torch.Tensor([eps]))


	def reset_parameters(self):
		reset(self.node_nn)
		reset(self.edge_nn)
		self.eps.data.fill_(self.initial_eps)

	def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
		if isinstance(x, Tensor):
			x: OptPairTensor = (x, x)

		# propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
		out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

		out = self.node_nn(out)
		#print(out.shape)
		return out

	def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
		edge_attr = self.edge_nn(edge_attr)
		return torch.cat((x_j, edge_attr), dim= -1)


	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.edge_nn,
                                   self.node_nn)