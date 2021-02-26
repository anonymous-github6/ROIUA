import os
import torch
import torch.utils.data as data

import numpy as np




class feature_qscore(data.Dataset):
	def __init__(self,root,label,transform = None):
		super(feature_qscore, self).__init__()
		self.root = root
		self.label_path = label
		self.transform = transform
		with open(self.label_path) as input_file:
			self.lines = input_file.readlines()
		
	def __getitem__(self,index):
		feature = np.zeros([512])
		line_vector = self.lines[index].strip().split(' ')
		for i in range(512):
			feature[i] = float(line_vector[i])
		#print(feature)
		return feature

	def __len__(self):
		return len(self.lines)
