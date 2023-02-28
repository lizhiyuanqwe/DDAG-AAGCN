import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# torch.set_printoptions(profile="full")


class Metric_Learning_based_Adjacency_Matrix(nn.Module):
	def __init__(self, in_features, rank=2048, alpha=1.0):
		super(Metric_Learning_based_Adjacency_Matrix, self).__init__()
		self.in_features = in_features   # 2048
		self.rank = rank
		self.alpha = alpha
		self.P = nn.Parameter(torch.zeros(size=(in_features, rank)))
		nn.init.xavier_uniform_(self.P.data, gain=1.414)

	def forward(self, features):
		assert len(features.shape) == 2, 'wrong shape of input features'
		M = torch.mm(self.P, self.P.T)  # 2048*2048
		bs = features.shape[0]			# bs:16 features:[16,2048]
		mahalanobis_dist = torch.autograd.Variable(torch.zeros(bs, bs)).cuda()	#[16,16]
		for i in range(0, bs):
			for j in range(i+1, bs):
				delta = (features[i, :]-features[j, :]).unsqueeze(-1)          	# x_i-x_j	#delta [2048,1]
				d_ij = torch.sqrt(torch.mm(torch.mm(delta.T, M), delta))       	# (x_i-x_j)^T M (x_i-x_j)
				mahalanobis_dist[i, j] = d_ij
		mahalanobis_dist = mahalanobis_dist + mahalanobis_dist.T				#[16,16]
		adjacency_pre = torch.exp(- self.alpha * mahalanobis_dist)				#[16,16]
		return self.__normalize_adjency(adjacency_pre), adjacency_pre

	def __normalize_adjency(self, adjacency):
		assert len(adjacency.shape) == 2, "wrong input of adjacency matrix"
		dregree_sqrt = torch.diag_embed(torch.pow(torch.sum(adjacency, dim=1), -0.5))  # D^-0.5
		return torch.mm(torch.mm(dregree_sqrt, adjacency), dregree_sqrt)               # D^-0.5*A*D^-0.5



class Attention_based_Adjacency_Matrix(nn.Module):
	def __init__(self, in_features, alpha=1.0):
		super(Attention_based_Adjacency_Matrix, self).__init__()
		self.in_features = in_features   # 2048
		self.alpha = alpha
		self.a = nn.Parameter(torch.zeros(size=([self.in_features, 1])))
		nn.init.constant_(self.a.data, 0.01)

	def forward(self, features):
		assert len(features.shape) == 2, 'wrong shape of input features'
		n, d = features.shape							# n:16  d:2048  features:[16,2048]
		f1 = features.unsqueeze(1).repeat(1, n, 1)		# f1:[16,16,2048]
		f2 = f1.permute(1, 0, 2)						# f2:[16,16,2048]
		fi_minus_fj = torch.abs(f1-f2)					# fi:[16,16,2048]
		score = torch.matmul(fi_minus_fj, self.a).squeeze()		# score:[16,16]
		adjacency = torch.exp(-self.alpha * F.relu(score))		# adjacency:[16,16]
		return self.__normalize_adjency(adjacency), adjacency

	def __normalize_adjency(self, adjacency):
		assert len(adjacency.shape) == 2, "wrong input of adjacency matrix"
		dregree_sqrt = torch.diag_embed(torch.pow(torch.sum(adjacency, dim=1), -0.5))  # D^-0.5
		return torch.mm(torch.mm(dregree_sqrt, adjacency), dregree_sqrt)               # D^-0.5*A*D^-0.5


