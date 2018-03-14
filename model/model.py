import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet18', 'resnet50', 'densenet161', 'MRCNN_resnet50',
           'MRCNN_densenet161','MRCNN_HardClass',
           'resnet18_places365','resnet50_places365','densenet161_places365']
model_urls = {
    'resnet18': '../resnet/whole_resnet18_places365.pth.tar',
    'resnet50': '../resnet/whole_resnet50_places365.pth.tar',
    'dense161': '../resnet/whole_densenet161_places365.pth.pth.tar',
}

def resnet18(pretrained=False, pretrained_model='', n_class = 80):

	model = torch.load(model_urls['resnet18'])
	model.fc = torch.nn.Linear(512,n_class)
	if pretrained:
		model.load_state_dict(torch.load(pretrained_model))

	return model

def resnet50(pretrained=False, pretrained_model='', n_class = 80):
	model = torch.load(model_urls['resnet50'])
	model.fc = torch.nn.Linear(2048,n_class)
	if pretrained:
		model.load_state_dict(torch.load(pretrained_model))

	return model

def MRCNN_HardClass(pretrained=False, pretrained_model='', n_class = 16):
	model = torch.load(model_urls['resnet50'])
	model.fc = torch.nn.Linear(2048, n_class)
	if pretrained:
		model.load_state_dict(torch.load(pretrained_model))

	return model

def resnet18_places365():
	model = torch.load(model_urls['resnet18'])
	return model

def resnet50_places365():
	model = torch.load(model_urls['resnet50'])
	return model

def dense161_places365():
	model = torch.load(model_urls['dense161'])
	return model