import sys
sys.path.append('/home/songjian/project/MSCD/github_code')
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy.ndimage import affine_transform
import utils.imutils as imutils

def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img



class OpenMapCDeDatset_BCD(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        # pre_img = one_hot_encoding(pre_img)
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'OSM', self.data_list[index] + '.png')
        post_path = os.path.join(self.dataset_path, 'OPT', self.data_list[index] + '.png')
        label_path = os.path.join(self.dataset_path, 'BC_GT', self.data_list[index] + '.png')
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        label = label - 1
        label[label == -1] = 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)



class OpenMapCDeDatset_SCD(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, pre_lc_label, post_lc_label, label):
        if aug:
            pre_img, post_img, pre_lc_label, post_lc_label, label = imutils.random_crop_multicd(pre_img, post_img, pre_lc_label, post_lc_label, label,
                                                                                 self.crop_size)
            pre_img, post_img, pre_lc_label, post_lc_label, label = imutils.random_fliplr_multicd(pre_img, post_img, pre_lc_label,post_lc_label,
                                                                                   label)
            pre_img, post_img, pre_lc_label, post_lc_label, label = imutils.random_flipud_multicd(pre_img, post_img, pre_lc_label,post_lc_label,
                                                                                   label)
            pre_img, post_img, pre_lc_label, post_lc_label, label = imutils.random_rot_multicd(pre_img, post_img, pre_lc_label, post_lc_label, label)

        # pre_img = one_hot_encoding(pre_img)
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, pre_lc_label, post_lc_label, label

    def __getitem__(self, index):
        pre_map_path = os.path.join(self.dataset_path, 'OSM', self.data_list[index] + '.png')
        post_path = os.path.join(self.dataset_path, 'OPT', self.data_list[index] + '.png')
        pre_label_path = os.path.join(self.dataset_path, 'LC_GT_OSM', self.data_list[index] + '.png')
        post_label_path = os.path.join(self.dataset_path, 'LC_GT_OPT', self.data_list[index] + '.png')
        label_path = os.path.join(self.dataset_path, 'SC_GT', self.data_list[index] + '.png')

        pre_img = self.loader(pre_map_path)
        post_img = self.loader(post_path)
        pre_lc_label = self.loader(pre_label_path)
        post_lc_label = self.loader(post_label_path)
        label = self.loader(label_path)
        label = label - 1
        label[label == -1] = 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, pre_lc_label, post_lc_label, label = self.__transforms(True, pre_img, post_img, pre_lc_label, post_lc_label, label)
        else:
            pre_img, post_img, pre_lc_label, post_lc_label, label = self.__transforms(False, pre_img, post_img, pre_lc_label, post_lc_label, label)
            label = np.asarray(label)
            pre_lc_label = np.asarray(pre_lc_label)
            post_lc_label = np.asarray(post_lc_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, pre_lc_label, post_lc_label, label, data_idx

    def __len__(self):
        return len(self.data_list)
