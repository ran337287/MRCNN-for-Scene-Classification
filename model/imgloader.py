import torch.utils.data as data
from PIL import Image
import os
import torch
import numpy as np
import json
__all__ =['ImageData','ImageData_softlabel','Test_ImageData']

##Class ImageData
##      return img, annot, img_path
##Class ImageData_softlabel
##      return img, annot,softlabel,img_path
##Class Test_ImageData
##		return img,img_path


class ImageData(data.Dataset):
    def __init__(self, data_root, data_list, transform=None, target_transform=None, multi_flag=False):
        self.root = data_root  # images dataroot
        self.transform = transform
        self.target_transform = target_transform
        self.multi_flag = multi_flag

        with open(data_list) as json_file:
            img_list = json.load(json_file)
        if multi_flag:
            for i,k in enumerate(img_list):
                annot = torch.FloatTensor(1)
                annot.resize_(80).fill_(0)
                for j, v in enumerate(k['label_id']):
                    annot[int(float(v))] = 0.9
                img_list[i]['label_id'] = annot

        self.img_paths = img_list
        self.n_data = np.max(len(self.img_paths))
        if not self._check_exists():
            raise RuntimeError('Dataset not found')

    def __getitem__(self, index):
        img_path = self.img_paths[index]['image_id']

        img = Image.open(self.root + img_path.strip('\r\n')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.multi_flag:
            annot = self.img_paths[index]['label_id']
            return img, annot, img_path
        else:
            annot = torch.FloatTensor([int(self.img_paths[index]['label_id'])])

        return img, annot, img_path

    def __len__(self):
        return self.n_data

    def _check_exists(self):
        return os.path.exists(self.root)

class ImageData_softlabel(data.Dataset):
    def __init__(self, data_root, data_list, transform=None, target_transform=None, multi_flag=False):
        self.root = data_root  # images dataroot
        self.transform = transform
        self.target_transform = target_transform
        self.multi_flag = multi_flag

        with open(data_list) as json_file:
            img_list = json.load(json_file)

        if multi_flag:
            for i,k in enumerate(img_list):
                annot = torch.FloatTensor(1)
                annot.resize_(80).fill_(0)
                for j, v in enumerate(k['label_id']):
                    annot[int(float(v))] = 0.9
                img_list[i]['label_id'] = annot

        self.img_paths = img_list
        self.n_data = np.max(len(self.img_paths))
        if not self._check_exists():
            raise RuntimeError('Dataset not found')

    def __getitem__(self, index):
        img_path = self.img_paths[index]['image_id']

        img = Image.open(self.root + img_path.strip('\r\n')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)


        if self.multi_flag:
            annot = self.img_paths[index]['label_id']
            return img, annot, img_path
        else:
            annot = torch.FloatTensor([int(self.img_paths[index]['label_id'])])
            softlabel = torch.FloatTensor([(self.img_paths[index]['softlabel'])])
            softlabel = softlabel.view(365)

        return img, annot,softlabel,img_path

    def __len__(self):
        return self.n_data

    def _check_exists(self):
        return os.path.exists(self.root)

class test_ImageData(data.Dataset):
	def __init__(self, data_root, data_list, transform=None, target_transform=None):
		self.root = data_root # images dataroot
		self.transform = transform
		self.target_transform = target_transform


		with open(data_list) as json_file:
			img_list = json.load(json_file)

		self.img_paths = img_list
		self.n_data = np.max(len(self.img_paths))
		if not self._check_exists():
			raise RuntimeError('Dataset not found')

	def __getitem__(self, index):
		img_path = self.img_paths[index][u'image_id']
		img = Image.open(self.root + img_path.strip('\r\n')).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		return img,img_path

	def __len__(self):
		return self.n_data

	def _check_exists(self):
		return os.path.exists(self.root)