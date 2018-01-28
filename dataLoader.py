#import os
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data
from skimage import io
#from PIL import Image

def imread(path):
    return io.imread(path)[:,:,0:3]


CLASS_NAME=['asian', 'western']

def readAll(label_file, shuffle=True):
    with open(label_file) as f:
        entries = f.read().split('\n')
    if shuffle:
        np.random.shuffle(entries)

    return entries


class ImageSet(data.Dataset):
    def __init__(self, entries):
        self.trans = transforms.ToTensor()
        labeled = []
        for line in entries:
            if not line.strip():
                continue
            img_path, label = line.split(',')
            img_path = img_path.strip()
            try:
                label = int(label.strip())
            except:
                print(type(label), label)
                raise
            labeled.append({'path':img_path, 'label':label})

        self.labeled = labeled

    def __getitem__(self, index):
        entry = self.labeled[index // 2]
        img_path = entry['path']
        label = entry['label']
        img = imread(img_path)
        if index % 2 == 0:
            img = img[:, ::-1, :].copy()
        #img = img.transpose((2,0,1))
        return self.trans(img), label

    def __len__(self):
        return len(self.labeled) * 2

    def getName(self):
        return CLASS_NAME
