
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.join(os.getcwd(), 'cffis'))
import random
import torch._utils
from PIL import Image
import pandas as pd
import h5py
from dataloaders import data_transform

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


class NYUD2Dataset(Dataset):

    def __init__(self, csv_file, root_dir, height, width, args, phase, transform=None):
        self.image_list = []
        self.label_list = []
        self.ref_list = []
        self.transform = transform
        self.height = height
        self.width = width
        self.args = args
        self.phase = phase
        self.rgbd_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.rgbd_frame)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_dir,
                                 self.rgbd_frame.iloc[idx, 0])
        rgb_h5, depth_h5 = self.load_h5(file_name)  # array

        rgb_image = Image.fromarray(rgb_h5, mode='RGB')
        depth_image = Image.fromarray(depth_h5.astype('float32'), mode='F')

        tRgb = data_transform.Compose([transforms.Resize(240, interpolation=Image.BILINEAR),
                                       transforms.CenterCrop((228, 304)),
                                       transforms.ToTensor()])
        tDepth = data_transform.Compose([transforms.Resize(240, interpolation=Image.NEAREST),
                                         transforms.CenterCrop((228, 304))])
        rgb_image = tRgb(rgb_image)
        depth_image = tDepth(depth_image)

        image = np.asarray(rgb_image)
        label = np.asarray(depth_image)

        sample = {'image': image, 'label': label}
        if self.transform:
           sample = self.transform(sample)

        return sample

    def parse_list(self, list_file):
        """
        parse the list
        """ 
        with open(list_file, 'r') as f:
            l = f.read().splitlines()
        self.image_list = [line.split()[0] for line in l]
        self.label_list = [line.split()[1] for line in l]

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename, 'r')
    #    print (f.keys())
        rgb = f['rgb'][:].transpose(1,2,0)
        depth = f['depth'][:]
        return (rgb, depth)



class Rotate(object):
    """Ramonly rotate the image and corresponding depth """
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, sample):
        image, label, ref = sample['image'], sample['label'], sample['ref']
        rnd_angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((self.width/2, self.height/2), rnd_angle, 1.0 )
        image = cv2.warpAffine(image, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT101)
        label = cv2.warpAffine(label, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT_101,flags=cv2.INTER_NEAREST)
        ref   = cv2.warpAffine(ref  , M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT_101,flags=cv2.INTER_NEAREST)
        sample = {'image':image, 'label':label, 'ref':ref}
        return sample

class Resize(object):
    """Ramonly resize the image and corresponding depth """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def __call__(self, sample):
        image, label, ref = sample['image'], sample['label'], sample['ref']
        rnd_scale = random.uniform(1., 1.2) 
        new_width = np.round(rnd_scale*self.width).astype('int')
        new_height = np.round(rnd_scale*self.height).astype('int')
        image = cv2.resize(image,(new_width, new_height))
        label = cv2.resize(label, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        ref   = cv2.resize(ref  , (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        sample = {'image':image, 'label':label, 'ref':ref}
        return sample

class Crop(object):
    """Crop the image and corresponding depth """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
    def __call__(self, sample):
        image, label, ref = sample['image'], sample['label'], sample['ref']
        new_height = image.shape[0]
        new_width  = image.shape[1]
        rnd_h = random.randint(0, new_height-self.height)
        rnd_w = random.randint(0, new_width-self.width)
        image = image[rnd_h:rnd_h+self.height, rnd_w:rnd_w+self.width, :]
        label = label[rnd_h:rnd_h+self.height, rnd_w:rnd_w+self.width]
        ref   = ref[rnd_h:rnd_h+self.height, rnd_w:rnd_w+self.width]
        sample = {'image':image, 'label':label, 'ref':ref}
        return sample

class Flip(object):
    """Ramonly flip the image and corresponding depth """
        
    def __call__(self, sample):
        image, label, ref ,y_mask= sample['image'], sample['label'], sample['ref'], sample['y_mask']
        rnd_flip = random.randint(0, 1)
        if rnd_flip == 1:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
            ref   = cv2.flip(ref, 1)
            y_mask = cv2.flip(y_mask,1)
        sample = {'image':image, 'label':label, 'ref':ref, 'y_mask':y_mask}
        return sample

class CorlorJitter(object):
    """Ramonly flip the image and corresponding depth """
        
    def __call__(self, sample):
        image, label, ref = sample['image'], sample['label'], sample['ref']
        image = image.astype('float')
        rnd_color = np.random.uniform(low=0.8, high=1.2, size=(3,))        
        image = image*rnd_color
        sample = {'image':image, 'label':label, 'ref':ref}
        return sample

class DownsampleGT(object):
    """Crop the image and corresponding depth """
    def __init__(self, width, height, residual,  record,val=False):
        #self.width = width / 2
        #self.height = height / 2
        self.width = width 
        self.height = height 
        self.residual = residual
        self.val = val

        self.record = record
        
    def __call__(self, sample):
        image, label, ref ,y_mask= sample['image'], sample['label'], sample['ref'], sample['y_mask']

        # valid mask
        mask = (label > 0).astype(np.uint8)

        # initialize the sign of the label
        label_sign = np.zeros(label.shape) 

        # downsample
        if self.residual==1:
            ref[ref<0] = 0
            ref[ref>80] = 80
            label = label - ref

        elif self.residual==2:
            ref[ref<0] = 0
            ref[ref>80] = 80 
            rel1 = np.divide(label, ref)
            rel2 = np.divide(ref, label)
            label = np.minimum(rel1, rel2)
            label_sign = (rel1 > 1).astype(np.uint8)

        if not self.val:
            label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            label_sign = cv2.resize(label_sign, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        sample = {'image':image, 'label':label, 'label_sign': label_sign, 'ref':ref, 'mask':mask, 'y_mask':y_mask}
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label, label_sign, ref, mask, y_mask = sample['image'], sample['label'], sample['label_sign'], sample['ref'], sample['mask'], sample['y_mask']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        sample = {'image':image.astype(np.float32), 
                  'label':label.astype(np.float32), 
                  'label_sign':label_sign.astype(np.uint8), 
                  'ref':ref.astype(np.float32), 
                  'mask':mask.astype(np.uint8),
                  'y_mask':y_mask.astype(np.float32)
                  }
        return sample

class Normalize(object):
    """Normalize image and ref to [-1, 1]"""
    def __init__(self, img_mean, img_std, ref_mean, ref_std):
        self.img_mean = img_mean
        self.img_std = img_std
        self.ref_mean = ref_mean
        self.ref_std = ref_std

    def __call__(self, sample):
        image, label, label_sign, ref, mask,y_mask = sample['image'], sample['label'], sample['label_sign'], sample['ref'], sample['mask'], sample['y_mask']
        image = (image.astype(np.float32)/255.0 - self.img_mean) / self.img_std
        ref = (ref - self.ref_mean) / self.ref_std
        sample = {'image':image, 'label':label, 'label_sign':label_sign, 'ref':ref, 'mask':mask, 'y_mask':y_mask}
        return sample

def get_loader_nyud2(csv_file, root_dir, batchsize, residual, args, phase='train'):
    height = 256
    width = 320
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    ref_mean = 3.01047486221
    ref_std  = 1.07943839977
    if phase=='train':
        composed = transforms.Compose([#Rotate(width, height), 
                                       #Resize(width, height),
                                       #Crop(width, height),
                                       Flip(),
                                       # CorlorJitter(),
                                       DownsampleGT(width, height, residual, args.record),
                                       Normalize(img_mean, img_std, ref_mean, ref_std),
                                       ToTensor()
                                       ])
        num_workers = 4
    else:
        composed = transforms.Compose([DownsampleGT(width, height, 0, args.record),
                                       Normalize(img_mean, img_std, ref_mean, ref_std),
                                       ToTensor()
                                       ])
        num_workers = 4
    
    nyud2 = NYUD2Dataset(csv_file,
                         root_dir, height, width, args, phase) #,
                         # transform = composed)


    return DataLoader(nyud2, batch_size=batchsize, shuffle=True, num_workers=num_workers)


def get_loader(dataset, *args):
    return eval('get_loader_%s' % dataset)(*args)


if __name__ == '__main__':
    pass
