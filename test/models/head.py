import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class ROIUA(nn.Module):
	def __init__(self, thresh  = 0.9):
		super(ROIUA, self).__init__()
		self.channel = 512
		self.fc_r1 = nn.Linear(512, 256)
		self.fc_r2 = nn.Linear(256, 1)
		self.relu = nn.ReLU(inplace = True)
		self.sigmoid = nn.Sigmoid()
		self.thresh = nn.Parameter(torch.FloatTensor(1).fill_(thresh))
		nn.init.normal_(self.fc_r1.weight, mean=0, std = 0.01)
		nn.init.constant_(self.fc_r1.bias, 0)
		nn.init.normal_(self.fc_r2.weight, mean=0, std = 0.01)
		nn.init.constant_(self.fc_r2.bias, 0)
	
	def forward(self, x):
		batch, d_f = x.size()
		s_out = self.fc_r1(x)
		s_out = self.relu(s_out)
		s_out = self.fc_r2(s_out)
		s_out = self.sigmoid(s_out)
		s_out_th = s_out - self.thresh
		if not self.training:
			return s_out
		return s_out, s_out_th

