import torch.utils.data
import json
import numpy as np
from torch.autograd import Variable
from model.imgloader import ImageData
import torchvision.transforms.functional as Fe
from torchvision import transforms
import torch.nn.functional as F
########## params ##########
net_mode = 'densenet161'
fc_nodes = 0
if net_mode=='resnet50':
	fc_nodes = 2048
if net_mode=='resnet18':
	fc_nodes = 512


train_data_root = '../data/ai_challenger_scene_train_20170904/'
train_data_path = train_data_root + 'scene_train_images_20170904/'
train_json = train_data_root + 'scene_train_annotations_20170904.json'

pre_trained_model_resnet = "../resnet/whole_{0}_places365.pth.tar".format(net_mode)
    # "../resnet/whole_resnet18_places365.pth.tar"
    # "../resnet/resnet18-5c106cde.pth"

n_class = 80
img_size = 256
BatchSize = 64
n_epoch = 180
cuda = True

############ load data ############

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
def crop_transform(img):
    imgs = Fe.ten_crop(img,224)
    return torch.stack([Fe.normalize(Fe.to_tensor(x), mean=mean, std=std) for x in imgs],0)

img_transform = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.Lambda(crop_transform)
    ])

dataset = ImageData(data_root=train_data_path, data_list=train_json,
                    transform=img_transform,multi_flag=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BatchSize, shuffle=False, num_workers=8)

my_model = torch.load(pre_trained_model_resnet)
print my_model
if cuda:
    my_model.cuda()
predict = []



for epoch in xrange(1):

    train_info = []
    data_iter = iter(data_loader)

    j = 0
    result = []

    while j < len(data_loader):
        data = data_iter.next()

        img, annot, img_path = data
        bs,ncrops,c,h,w = img.size()
        img = (img.view(-1,c,h,w))


        input_img = Variable(img.cuda(), volatile=True)
        pred_pro = my_model(input_img)
        result = F.softmax(pred_pro)

        pred_pro = result.view(bs,ncrops,-1).mean(1)
        pred_pro = pred_pro.cpu().data.numpy()

        for k, pred in enumerate(pred_pro):
            temp ={u'image_id': [], u'label_id': [], u'softlabel':[]}
            predict_label = np.argsort(-pred)
            temp[u'image_id'] = img_path[k]
            temp[u'label_id'] = int(annot[k].numpy()[0])
            temp[u'softlabel'] = pred.tolist()
            # print temp

            train_info.append(temp)

        j += 1

    f = open(train_data_root +'train_data_densenet_softlabel.json', 'w')
    f.write(json.dumps(train_info))
    f.close()






