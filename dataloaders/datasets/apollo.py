import os
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils import data
import pandas as pd
from dataloaders.utils_zzm import encode_segmap_gray_apollo
from torchvision import transforms
from dataloaders import custom_transforms as tr

#两种生成器 方式1:通过Dataset类和Dataloader类 方式2:通过yield生成generator
#方式1
class ApolloSegmentation(data.Dataset):
    NUM_CLASSES = 8

    def __init__(self, args,  csv_file, mode='train'):#没加transform
        super(ApolloSegmentation, self).__init__()
        self.args = args
        self.data = pd.read_csv(os.path.join(os.getcwd(), "data_list", csv_file), header=None,names=["image","label"])
        self.mode = mode
        self.images = self.data["image"].values[1:]
        self.labels = self.data["label"].values[1:]

    def __len__(self):
        return len(self.data)-1

    def __getitem__(self, idx):


        ori_image = np.array(Image.open(self.images[idx]).convert('RGB'))
        ori_mask = np.array(Image.open(self.labels[idx]))
        train_img, train_mask = tr.crop_resize_data(ori_image, ori_mask)
        # Encode
        train_mask = encode_segmap_gray_apollo(train_mask)

        _img = Image.fromarray(train_img)
        _target = Image.fromarray(train_mask)

        sample = {'image':_img, 'label':_target}

        #transform
        if self.mode == 'train':
            return self.transform_tr(sample)
        elif self.mode == 'val':
            return self.transform_val(sample)
        elif self.mode == 'test':
            return self.transform_ts(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)




#方式2

def transform_tr(sample):
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        # tr.RandomGaussianBlur(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    return composed_transforms(sample)

def transform_val(sample):
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    return composed_transforms(sample)

def transform_ts(sample):
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    return composed_transforms(sample)

def apollo_image_gen(csv_file, mode='train', batch_size=4, image_size=[1024, 384], crop_offset=690):
    # Arrange all indexes
    data_list = pd.read_csv(os.path.join(os.getcwd(), "data_list", csv_file), header=None,names=["image","label"])
    all_batches_index = np.arange(0, len(data_list)-1)
    # print("train data read, length:", len(data_list))
    out_samples = {'image': [], 'label': []}
    images = data_list["image"].values[1:]
    labels = data_list["label"].values[1:]

    while True:
        # np.random.shuffle(all_batches_index)
        for idx in all_batches_index:
            # ori_image = np.array(Image.open(images[idx]).convert('RGB'))
            # ori_mask = np.array(Image.open(labels[idx]))
            ori_image_bgr = cv2.imread(images[idx])
            ori_image = ori_image_bgr[:, :, ::-1]
            ori_mask = cv2.imread(labels[idx], cv2.IMREAD_GRAYSCALE)
            train_img, train_mask = tr.crop_resize_data(ori_image, ori_mask, image_size, crop_offset)
            #encode
            train_mask = encode_segmap_gray_apollo(train_mask)
            # _img = Image.fromarray(train_img)
            # _target = Image.fromarray(train_mask)
            # sample = {'image': _img, 'label': _target}
            sample = {'image': train_img, 'label': train_mask}
            if mode == 'train':
                sample = transform_tr(sample)
            elif  mode == 'val':
                sample = transform_val(sample)
            elif mode == 'test':
                sample = transform_ts(sample)

            out_samples['image'].append(sample['image'])
            out_samples['label'].append(sample['label'])
            # 至此, out_samples['image']是list of tensor

            # print("idx:", idx)
            # print(sample['image'].size)
            # print(len(out_samples['image']))
            if len(out_samples['image']) >= batch_size:
                #tensor要转成array 再转成tensor
                #解构
                out_samples['image'] = torch.from_numpy(np.array(out_samples['image']))
                out_samples['label'] = torch.from_numpy(np.array(out_samples['label']))
                yield out_samples
                out_samples = {'image': [], 'label': []}








if __name__ == '__main__':
    from dataloaders.utils_zzm import decode_segmap_gray_apollo, decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    apollo_train = ApolloSegmentation("train_lite.csv", mode="train")
    dataloader = DataLoader(apollo_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            # segmap = decode_segmap_gray_apollo(tmp)
            segmap = decode_segmap(tmp, 'apollo')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)