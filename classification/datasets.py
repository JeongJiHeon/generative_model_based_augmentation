import os
import sys
import numpy as np
import torch
import random
import glob
import pandas as pd

from torch.utils.data import Dataset

import torchvision
# import torchvision.transforms as T

from config import parse_arguments
from PIL import Image
import cv2
import SimpleITK as sitk
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

make_label = lambda path : [0] if 'mutant' in path.lower() else [1] # 0 : mutant, 1 : wild
make_real  = lambda path : True if 'Glioma_AMC_GBM_TCGA_Final_DataSet'.lower() in path.lower() else False


train_transform = A.Compose([
    #A.CoarseDropout(p=0.25, min_holes = 2, max_holes = 8, max_height=16, max_width=16, min_height=4, min_width=4),
    A.ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2, rotate_limit=30, p=0.25),
    ToTensorV2(),
                ])

test_transform = A.Compose([
                    ToTensorV2(),
                ])



def Normalization_MeanStd(tmp_val_img):
    tmp_val_img = np.rot90(tmp_val_img)
    #tmp_val_img = resize(tmp_val_img, (256, 256), interpolation = INTER_NEAREST)
    #tmp_val_img = tmp_val_img.reshape(256,256)
    if tmp_val_img.min() < 0: 
        tmp_val_img = tmp_val_img + abs(tmp_val_img.min())
    #MAX = np.percentile(tmp_val_img.flatten(),98)
    #print(MAX , tmp_val_img.max())
    
    #tmp_val_img[tmp_val_img > MAX] = MAX
    #tmp_val_img = (tmp_val_img/MAX)*255 # scale to be 0 to 255 (uint8)

    tmp_val_img = (tmp_val_img/tmp_val_img.max())*255 # scale to be 0 to 255 (uint8)

    tmp_val_img = tmp_val_img.astype(np.uint8)

    img_mean = tmp_val_img.mean() # normalization
    img_std = tmp_val_img.std()
    if(img_std != 0): tmp_val_img = (tmp_val_img - img_mean) / img_std
    else: tmp_val_img = (tmp_val_img - img_mean) 

    return tmp_val_img

class DiseaseDataset(Dataset):
    def __init__(self, args, image_size=256, transform=None, test = False, mode = 'train', clinical = 'no'):
        self.args = args
        self.image_size = image_size
        self.samples = []
        self.transform = train_transform if (mode == 'train') and (args.augment) else test_transform
        
        self.csv = pd.read_csv(f'csv/{mode}_real.csv')
        if mode == 'train':
            if clinical == 'no':
                for i in range(args.fake_slice):
                    self.csv = self.csv.append(pd.read_csv(f'csv/fake_{i:03}.csv'))
                    
            elif clinical == 'large':
                for i in range(args.fake_slice):
                    self.csv = self.csv.append(pd.read_csv(f'csv/clinical/large/fake_{i:03}_large.csv'))
                
            elif clinical == 'small':
                for i in range(args.fake_slice):
                    self.csv = self.csv.append(pd.read_csv(f'csv/clinical/small/fake_{i:03}_small.csv'))
                
            elif clinical == 'CE':
                for i in range(args.fake_slice):
                    self.csv = self.csv.append(pd.read_csv(f'csv/clinical/CE/fake_{i:03}_CE.csv'))
                    
            elif clinical == 'noCE':
                for i in range(args.fake_slice):
                    self.csv = self.csv.append(pd.read_csv(f'csv/clinical/noCE/fake_{i:03}_noCE.csv'))
                
                
                
                
        print('mode', mode, 'total:', len(self.csv))

    def __getitem__(self, idx):
        
        samples = self.csv.iloc[idx]
        imgs = self._preprocessing(samples['path'].replace('/mnt', self.args.original_path))
        labels = np.array([samples['mutation']]).astype(np.float32)
        
        return imgs, labels
            
    def __len__(self):
        return len(self.csv)
    
    def _preprocessing(self, path):
        
        image = np.zeros((self.image_size, self.image_size, 2)).astype(np.float32)
        img = np.load(path)[:,:,:2]
        
        image[:,:,0] = self._standardization(img[:,:,0])
        image[:,:,1] = self._standardization(img[:,:,1])
        
        image = self.transform(image = image)['image']
        

        return image

    
    def _standardization(self, img):        
        norm_img = Normalization_MeanStd(img)        
        return norm_img




# class DiseaseDataset(Dataset):
#     def __init__(self, path_list, args, image_size=256, transform=None, test = False, mode = 'train'):
#         self.args = args
#         self.image_size = image_size
#         self.samples = []
#         self.transform = train_transform if (mode == 'train') and (args.augment) else test_transform
        
#         if mode == 'train':
#             for path in path_list:
#                 print(path)

        
#         if mode == 'train':
#             if args.real_ratio != 1:
#                 total = args.wild
#                 mutant_total = 0
#                 wild_total = 0
#                 for path in path_list:
#                     sample_list = glob.glob(os.path.join(path, '*'))
#                     if not make_real(sample_list[0]): # Fake
#                         fake_number = total - mutant_total if 'mutant' in sample_list[0].lower() else total - wild_total
#                         sample_list = [ {'imgs' : sample, 'labels' : make_label(sample), 'real' : make_real(sample) } for sample in sample_list ][:fake_number]

#                     else: # Real
#                         real_number = int(len(sample_list) * args.real_ratio) if 'mutant' in sample_list[0].lower() else int(len(sample_list) * args.real_ratio)
                        
#                         if 'mutant' in sample_list[0].lower():
#                             mutant_total += real_number
#                         else:
#                             wild_total += wild_total
                        
#                         sample_list = [ {'imgs' : sample, 'labels' : make_label(sample), 'real' : make_real(sample) } for sample in sample_list ][:real_number]

#                     self.samples += sample_list

#             elif args.real_ratio == 1:
#                 if args.no_add:
#                     args.mutant = args.wild
#                 total = int((args.fake_ratio+1) * args.wild) # Wild 전체 갯수

#                 for path in path_list:
#                     sample_list = glob.glob(os.path.join(path, '*'))
#                     if not make_real(sample_list[0]): # Fake
#                         fake_number = total - args.mutant if 'mutant' in sample_list[0].lower() else total - args.wild
#                         sample_list = [ {'imgs' : sample, 'labels' : make_label(sample), 'real' : make_real(sample) } for sample in sample_list ][:fake_number]
#                     else: # Real
#                         sample_list = [ {'imgs' : sample, 'labels' : make_label(sample), 'real' : make_real(sample) } for sample in sample_list ]

#                     self.samples += sample_list
#         else:
#             if args.no_add:
#                 args.mutant = args.wild
#             total = int((args.fake_ratio+1) * args.wild) # Wild 전체 갯수

#             for path in path_list:
#                 sample_list = glob.glob(os.path.join(path, '*'))
#                 if not make_real(sample_list[0]): # Fake
#                     fake_number = total - args.mutant if 'mutant' in sample_list[0].lower() else total - args.wild

#                     sample_list = [ {'imgs' : sample, 'labels' : make_label(sample), 'real' : make_real(sample) } for sample in sample_list ][:fake_number]
#                 else: # Real
#                     sample_list = [ {'imgs' : sample, 'labels' : make_label(sample), 'real' : make_real(sample) } for sample in sample_list ]

#                 self.samples += sample_list


#     def __getitem__(self, idx):
        
#         samples = self.samples[idx]
        
#         imgs = self._preprocessing(samples['imgs'], samples['real'])
#         labels = np.array(samples['labels']).astype(np.float32)
        
#         return imgs, labels
            
#     def __len__(self):
#         return len(self.samples)
    
#     def _preprocessing(self, path, real):
        
#         image = np.zeros((self.image_size, self.image_size, 2)).astype(np.float32)
        
#         img = np.load(path)[:,:,:2] if real else np.load(path)
        
#         image[:,:,0] = self._standardization(img[:,:,0])
#         image[:,:,1] = self._standardization(img[:,:,1])
        
#         image = self.transform(image = image)['image']
        

#         return image

    
#     def _standardization(self, img):        
#         norm_img = Normalization_MeanStd(img)        
#         return norm_img

# For test
if __name__ == '__main__':
    dataset = DiseaseDataset('./json/dummy.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

    for imgs, labels in iter(dataloader):
        print(imgs.shape, labels.shape)
        
