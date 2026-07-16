r""" ISIC few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import pandas as pd


class DatasetISIC(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize=False, seed=0, num_test_samples=0):
        self.split = split
        self.fold = fold
        self.benchmark = 'isic'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.seed = seed
        self.num_test_samples = num_test_samples

        self.base_path = os.path.join(datapath, 'ISIC')
        self.categories = ['1', '2', '3']
        self.category_names = ['nevus', 'melanoma', 'seborrheic_keratosis']

        self.class_ids = range(0, 3)
        self.cls_dict = {'nevus': "1", 'melanoma': "2", 'seborrheic_keratosis': "3"}
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

        self.transform = transform

    def __len__(self):
        if self.split == 'trn':
            return len(self.img_metadata)
        else:
            return self.num_test_samples if self.num_test_samples != 0 else 1000

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)

        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        category = self.category_names[class_sample]
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'neg_query_mask': torch.zeros_like(query_mask),
                 'query_name': query_name,
                 'query_img_path': os.path.join(self.base_path, query_name),
                 'org_query_imsize': org_qry_imsize,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_img_paths': [os.path.join(self.base_path, name) for name in support_names],
                 'class_id': torch.tensor(class_sample),
                 'category': category
                 }

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        ann_path = os.path.join(self.base_path, 'ISIC2018_Task1_Training_GroundTruth')
        query_mask_name = os.path.join(ann_path, query_id) + '_segmentation.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_mask_names = [os.path.join(ann_path, sid) + '_segmentation.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_mask_name)
        support_masks = [self.read_mask(name) for name in support_mask_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(list(self.class_ids))
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break

        return query_name, support_names, class_id

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted(glob.glob(os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input', cat, '*.jpg')))
            img_metadata.extend(img_paths)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        class_ids_df = pd.read_csv("datasets/isic/class_id.csv")
        img_paths = sorted(glob.glob(os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input', '*', '*.jpg')))
        for img_path in img_paths:
            basename = os.path.basename(img_path)
            if basename.endswith('.jpg'):
                img_metadata_classwise[
                    self.cls_dict[class_ids_df.loc[class_ids_df["ID"] == basename.split('.')[0], "Class"].values[0]]
                ] += [img_path]
        return img_metadata_classwise
