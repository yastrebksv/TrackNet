from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import math
import numpy as np

class trackNetDataset(Dataset):
    def __init__(self, mode, input_height=360, input_width=640):
        self.path_dataset = './datasets/trackNet'
        assert mode in ['train', 'val'], 'incorrect mode'
        self.data = pd.read_csv(os.path.join(self.path_dataset, 'labels_{}.csv'.format(mode)))
        print('mode = {}, samples = {}'.format(mode, self.data.shape[0]))         
        self.height = input_height
        self.width = input_width
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        path, path_prev, path_preprev, path_gt, x, y, status, vis = self.data.loc[idx, :]
        
        path = os.path.join(self.path_dataset, path)
        path_prev = os.path.join(self.path_dataset, path_prev)
        path_preprev = os.path.join(self.path_dataset, path_preprev)
        path_gt = os.path.join(self.path_dataset, path_gt)
        if math.isnan(x):
            x = -1
            y = -1
        
        inputs = self.get_input(path, path_prev, path_preprev)
        outputs = self.get_output(path_gt)
        
        return inputs, outputs, x, y, vis
    
    def get_output(self, path_gt):
        img = cv2.imread(path_gt)
        img = cv2.resize(img, (self.width, self.height))
        img = img[:, :, 0]
        img = np.reshape(img, (self.width * self.height))
        return img
        
    def get_input(self, path, path_prev, path_preprev):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))

        img_prev = cv2.imread(path_prev)
        img_prev = cv2.resize(img_prev, (self.width, self.height))
        
        img_preprev = cv2.imread(path_preprev)
        img_preprev = cv2.resize(img_preprev, (self.width, self.height))
        
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0

        imgs = np.rollaxis(imgs, 2, 0)
        return imgs
