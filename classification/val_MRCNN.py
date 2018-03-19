import model.model as model
import torch
import random
import json
import model.imgloader as imgloader
from torchvision import transforms
import numpy as np
import model.utils as utils
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as Fe
##################
manualSeed = random.randint(1, 10000)
print "Random Seed: %d" %(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

################# params ###########
cuda = True
img_size = 256
crop_img_size = 224
BatchSize = 64
fc_ori_lr_weight = 10
lr = 1e-4
n_epoch = 180

# loss_bce = torch.nn.MultiLabelSoftMarginLoss()
# loss_coeff = torch.FloatTensor([0.5])

n_class = 80
n_class_extra = 365# 0
##if has extra network n_class_extra = 365,else n_class_extra = 0

mode = 'valid'
model_root = '../xq_model/'
model_name = 'resnet50'
model_path = '171125_00_0_exlr30_MRCNN_densenet_56.pth'
pre_trained_model = model_root + model_path
my_model = model.resnet50(False,pre_trained_model,n_class+n_class_extra)

save_json = '../xq_result/json/{0}_{1}.json'.format(model_path,mode)
if cuda:
	my_model.cuda()
	
#########################################

######## define transform #############
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
img_transform_train= transforms.Compose([
	transforms.Resize(size=img_size),
	transforms.RandomSizedCrop(crop_img_size),
	transforms.RandomHorizontalFlip(),
	transforms.ColorJitter(0.4, 0.4, 0.4),
	transforms.ToTensor(),
	transforms.Normalize(mean=mean, std=std)
])

def ten_crop(img):
    imgs = Fe.ten_crop(img,crop_img_size)
    return torch.stack([Fe.normalize(Fe.to_tensor(x), mean=mean, std=std) for x in imgs],0)
img_transform_test = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        transforms.Lambda(ten_crop)
    ])

img_transform ={
	'train': img_transform_train,
	'valid': img_transform_test,
	'test': img_transform_test
}
################ data_root ########################
train_data_root = '../data/ai_challenger_scene_train_20170904/'
valid_data_root = '../data/ai_challenger_scene_validation_20170908/'
test_data_root = '../data/ai_challenger_scene_test_a_20170922/'

data_path={
	'train':train_data_root + 'scene_train_images_20170904/',
	'valid':valid_data_root + 'scene_validation_images_20170908/',
	'test':test_data_root + 'scene_test_a_images_20170922/'
}
data_json={
	'train':train_data_root + 'scene_train_annotations_20170904.json',
	'valid':valid_data_root + 'scene_validation_annotations_20170908.json',
	'test':test_data_root + 'scene_test_a_images_20170922.json'
}
ImageData = imgloader.ImageData
dataset = ImageData(data_root=data_path[mode], data_list=data_json[mode],
	                    transform=img_transform[mode])
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BatchSize,
											  shuffle=False, num_workers=10)
############### data_loader ####################


for epoch in xrange(1):
	my_model = my_model.eval()
	valid_info = []
	data_iter = iter(data_loader)

	j = 0
	result = []

	while j < len(data_loader):
		data = data_iter.next()
		if mode == 'valid':
			img, _, img_path = data
		if mode == 'test':
			img, img_path = data
		bs, ncrops, c, h, w = img.size()
		img = (img.view(-1, c, h, w))

		input_img = Variable(img.cuda(), volatile=True)
		pred_pro = my_model(input_img)
		result = F.softmax(pred_pro[:, n_class_extra:])

		pred_pro = result.view(bs, ncrops, -1).mean(1)
		pred_pro = pred_pro.cpu().data.numpy()

		for k, pred in enumerate(pred_pro):
			# print pred
			# print img_path[k]
			temp = {u'image_id': [], u'label_id': []}
			predict_label = np.argsort(-pred)[:3]
			#predict_label = utils.LabelMap(predict_label, 'inverted_map')
			temp[u'image_id'] = img_path[k]
			temp[u'label_id'] = predict_label.tolist()
			valid_info.append(temp)
		j += 1

	f = open(save_json, 'w')
	f.write(json.dumps(valid_info))
	f.close()

	os.system('python ../evaluation/scene_eval.py --submit ' + save_json + ' --ref ' +
	          data_json[mode])
	
