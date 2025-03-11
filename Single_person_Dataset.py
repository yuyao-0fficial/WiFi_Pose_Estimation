# author: Schwarzer_land

import os

import cv2
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset


class Single_Person_Dataset(Dataset):
    def __init__(self, root, train=1):
        """
        :param root: 路径，比如.\data
        :param train: 加载的数据集为训练还是测试集，默认为训练集
        """
        super(Single_Person_Dataset, self).__init__()
        # 获取基础路径
        if train == 1:
            self.flag = 'train'
        elif train == 0:
            self.flag = 'val'
        else:
            self.flag = 'test'
        self.CSI_root = os.path.join(root, 'csi', self.flag)  # .\single person\csi\train
        self.HM_root = os.path.join(root, 'heatmap', self.flag)  # .\single person\csi\train
        self.JPEG_root = os.path.join(root, 'jpeg', self.flag)  # .\single person\csi\train
        # 获取所需的路径
        self.csi_name = [i for i in os.listdir(self.CSI_root)]  # '00000001.mat'
        self.csi_list = [os.path.join(self.CSI_root, i) for i in
                         self.csi_name]  # .\single person\csi\training\00000001.mat
        self.csi_list.sort(key=lambda x: int(x[-12:-4]))
        self.jpeg_name = [i for i in os.listdir(self.JPEG_root)]
        self.jpeg_list = [os.path.join(self.JPEG_root, i) for i in self.jpeg_name]
        self.jpeg_list.sort(key=lambda x: int(x[-12:-4]))
        self.hm_name = [i for i in os.listdir(self.HM_root)]  # '00000001.mat'
        self.hm_list = [os.path.join(self.HM_root, i) for i in
                        self.hm_name]  # .\single person\csi\training\00000001.mat
        self.hm_list.sort(key=lambda x: int(x[-12:-4]))

    def __len__(self):
        return len(self.csi_list)

    def __getitem__(self, idx):
        csi_anno = scio.loadmat(self.csi_list[idx])['csi_anno']
        csi_anno_r = (np.swapaxes(csi_anno.reshape(27, 90, 5), 1, 2)).reshape(-1, 90)  # 135@90
        csi = np.append(np.append(csi_anno_r, np.zeros([135, 6]), axis=1), np.zeros([25, 96]), axis=0)  # 160@96
        # csi = np.append(np.real(csi), np.imag(csi), 0)
        csi = csi - csi.min()
        csi = csi / np.abs(csi).max()

        heatmap = scio.loadmat(self.hm_list[idx])['heatmap']['hm'][0, 0]
        # heatmap = heatmap/np.abs(heatmap).max()

        # heatmap_all = np.zeros([36, 64])
        depth = scio.loadmat(self.hm_list[idx])['heatmap']['depth'][0, 0]
        # heatmap_all = heatmap_all/heatmap_all.max()

        img = cv2.imread(self.jpeg_list[idx])
        # img_show = mplimg.imread(self.jpeg_list[idx])
        # plt.figure(0)
        # plt.imshow(img_show)
        # frame_2 = plt.figure(1)
        # fig_2 = frame_2.add_subplot(111)
        # fig_2.imshow(heatmap_all/255)
        # plt.show()

        return csi, heatmap, depth, img
