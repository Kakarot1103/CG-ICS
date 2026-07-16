r""" iSAID-5i few-shot semantic segmentation dataset """
import os

import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from .pascal import DatasetPASCAL


class DatasetISAID(DatasetPASCAL):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, seed=0, num_test_samples=0):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'isaid'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.seed = seed
        self.num_test_samples = num_test_samples

        datapath = os.path.join(datapath, 'iSAID')

        if self.split == 'trn':
            self.img_path = os.path.join(datapath, 'train/images')
            self.ann_path = os.path.join(datapath, 'train/semantic_png')
        else:
            self.img_path = os.path.join(datapath, 'val/images')
            self.ann_path = os.path.join(datapath, 'val/semantic_png')

        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.cats = [str(i) for i in self.class_ids]
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        if self.split == 'trn':
            return len(self.img_metadata)
        else:
            return self.num_test_samples if self.num_test_samples != 0 else 1000

    def read_mask(self, img_name, class_sample=None):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.png')

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('datasets/splits/isaid/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata
