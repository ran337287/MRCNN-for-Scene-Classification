import model.model as model
import torch
import random
import model.imgloader as  imgloader
from torchvision import transforms
import model.utils as utils
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
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
BatchSize = 1
fc_ori_lr_weight = 10
lr = 1e-4
n_epoch = 180

loss_bce = torch.nn.MultiLabelSoftMarginLoss()
loss_coeff = torch.FloatTensor([0.5])

n_class = 80
n_class_extra = 365 # 0
##if has extra network n_class_extra = 365,else n_class_extra = 0

mode = 'train'
model_root = '../xq_model/'
model_name = 'resnet50'
pre_trained_model = ''
my_model = model.resnet50(False,pre_trained_model,n_class+n_class_extra)

if cuda:
	my_model.cuda()
	loss_bce.cuda()

BASE_LR = lr
EPOCH_DECAY = 30
DECAY_WEIGHT = 0.1
optimizer = optim.RMSprop([{'params': my_model.fc.parameters(), 'lr': fc_ori_lr_weight * lr}],
						  lr=lr, weight_decay=1e-5)
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
	'train':train_data_root + 'train_data_densenet_softlabel.json',
	'valid':valid_data_root + 'scene_validation_annotations_20170908.json',
	'test':test_data_root + 'scene_test_a_images_20170922.json'
}
ImageData = imgloader.ImageData_softlabel
dataset = ImageData(data_root=data_path[mode], data_list=data_json[mode],
	                    transform=img_transform[mode])
############### data_loader ####################

for epoch in range(n_epoch):
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=BatchSize,
											  shuffle=True, num_workers=10)
	data_iter = iter(data_loader)
	i = 0
	t = 0

	optimizer = utils.exp_lr_sheduler(optimizer,epoch,BASE_LR,EPOCH_DECAY,DECAY_WEIGHT)
	train_info = []

	while i < len(data_loader):

		data = data_iter.next()
		img, annot, softlabel, img_path = data

		annot_temp = annot.numpy()

		annot = utils.gen_one_hot(annot, n_class)  # label to one-hot code

		if cuda:
			annot = annot.cuda()
			softlabel = softlabel.cuda()
			img = img.cuda()

		my_model.zero_grad()
		batch_size = annot.size(0)

		input_annot = torch.FloatTensor(batch_size, n_class)
		input_softlabel = torch.FloatTensor(batch_size, n_class_extra)
		input_img = torch.FloatTensor(batch_size, 3, img_size, img_size)
		if cuda:
			input_softlabel = input_softlabel.cuda()
			input_annot = input_annot.cuda()
			input_img = input_img.cuda()

		input_softlabel.resize_as_(softlabel).copy_(softlabel)
		input_annot.resize_as_(annot).copy_(annot)
		input_img = input_img.resize_as_(img).copy_(img)
		input_softlabel = Variable(input_softlabel)
		inputv_annot = Variable(input_annot)
		inputv_img = Variable(input_img)

		pred_pro_coarse = my_model(inputv_img)

		pred_pro_softcodes, pred_pro_hardcodes = pred_pro_coarse[:, :n_class_extra], pred_pro_coarse[:, n_class_extra:]

		err_softbce = torch.mean(loss_bce(pred_pro_softcodes, input_softlabel))
		err_hardbce = torch.mean(loss_bce(pred_pro_hardcodes, inputv_annot))
		err_softbce.backward(torch.FloatTensor([0.5]).cuda(), retain_graph=True)
		err_hardbce.backward()
		optimizer.step()

		pred_pro = pred_pro_hardcodes.cpu().data.numpy()
		batch_acc = 0.0

		for k, pred in enumerate(pred_pro):
			temp = {u'image_id': [], u'label_id': []}
			predict_label = np.argsort(-pred)[:3]
			temp[u'image_id'] = img_path[k]
			temp[u'label_id'] = predict_label.tolist()
			train_info.append(temp)

			if annot_temp[k] in predict_label.tolist():
				batch_acc += 1
			# t += 1
		print 'iter:%d/%d, batch_acc:%f' % (i, len(data_loader), batch_acc / batch_size)

		i += 1
	print 'epoch: %d/%d, err_bce: %f/%f' % (epoch, n_epoch,
	                                        err_hardbce.cpu().data.numpy(),
	                                        err_softbce.cpu().data.numpy())
	torch.save(my_model.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(model_root, model_name, epoch))
# if epoch%10 ==0:
# 	f = open(train_model + str(epoch) + '_train.json', 'w')
# 	f.write(json.dumps(train_info))
# 	f.close()
# 	os.system('python ../evaluation/scene_eval.py --submit ' + train_model + str(epoch) + '_train.json --ref ' +
# 			  train_json)

print 'done'


