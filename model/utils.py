import torch
def exp_lr_sheduler(optimizer, epoch, init_lr = 1e-4,lr_decay_epoch=30,DECAY_WEIGHT = 0.1):
	lr = init_lr *(DECAY_WEIGHT** (epoch//lr_decay_epoch))
	if epoch % lr_decay_epoch == 0:
		print ('LR is set to {}'.format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return optimizer

def gen_one_hot(labels, nclass):
	true_label = torch.FloatTensor(1)
	true_label.resize_(labels.size(0), nclass).fill_(0)
	for i,v in enumerate(labels.numpy()):
		true_label[i, v[0]] = 0.9
	return true_label

import numpy as np
##classifying 16 classes
def LabelMap(oriLabel_dict, mode):
	hardLabel = [5, 6, 24, 32, 33, 35, 38, 41, 45, 49, 58, 63, 64, 65, 67]
	#positive_label=[1,2,3,4,...,15]
	#negetive_label=[0]

	oriLabel_dict = oriLabel_dict.tolist()
	newLabel_dict = []
	if mode == 'norm_map':
		for i in oriLabel_dict:
			if int(i[0]) not in hardLabel:
				newLabel_dict.append(0)
			else:
				newLabel_dict.append(hardLabel.index(int(i[0]))+1)
		newLabel_dict = torch.FloatTensor(np.array(newLabel_dict))

	elif mode == 'inverted_map':
		for i in oriLabel_dict:
			if int(i) == 0:
				newLabel_dict.append(-1)
			else:
				newLabel_dict.append(hardLabel[int(i)-1])
		newLabel_dict = np.array(newLabel_dict)


	return newLabel_dict
def PredMap(oriPred_dict):
	hardLabel = [5, 6, 24, 32, 33, 35, 38, 41, 45, 49, 58, 63, 64, 65, 67]
	pred = oriPred_dict.tolist()

	softlabel = np.zeros(80)
	avg_class = pred[0]/(80-len(hardLabel))
	for j in range(0,80):
		if j in hardLabel:
			softlabel[j] = pred[hardLabel.index(j)+1]
		else:
			softlabel[j] = avg_class

	return softlabel
