from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import os
import random
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt
import torch


class Cardataset(Dataset):
    def __init__(self, root_dir, image_folder, target_folder, phase, n_classes=2,
                 transforms=transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])):
        self.root_dir = root_dir
        self.image_folder = image_folder
        self.target_folder = target_folder
        self.transforms = transforms
        self.image_names = sorted(glob.glob(os.path.join(root_dir, image_folder, '*')))
        self.mask_names = sorted(glob.glob((os.path.join(root_dir, target_folder, '*'))))
        if phase == 'train':
            self.flip = True
        else:
            self.flip = False
        self.n_class = n_classes

    def transform(self, image, mask):
        resize = transforms.Resize(size=(640, 1024))
        image = resize(image)
        mask = resize(mask)
        # m=random.random()
        # vertical anbd horizontal flipping of image and mask
        if self.flip:
            if random.random() > 0.5:
                image = tf.hflip(image)
                mask = tf.hflip(mask)
            if random.random() < 0.5:
                image = tf.vflip(image)
                mask = tf.vflip(mask)
        image = np.asarray(image)
        mask = np.asarray(mask)
        mask = mask / 255
        ones = np.count_nonzero(mask > 0)
        h, w = mask.shape
        zeros = (h * w) - ones
        if ones != 0:
            ratio = zeros / ones
        else:
            ratio = 1
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][mask == c] = 1
        try:

            image = transforms.ToTensor()(image)
        except Exception as e:
            print(e)
        # mask=transforms.ToTensor()(mask)
        #   print (type(mask))
        return image, target, mask, ratio

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        try:
            image = (Image.open(self.image_names[idx]))


        except Exception as e:
            print(e)
        mask = (Image.open(self.mask_names[idx]))
        image_name = self.mask_names[idx]
        # print(image_name)

        x, y, z, ratio = self.transform(image, mask)
        if self.transforms is not None:
            x = self.transforms(x)
        sample = {'image': x, 'mask': y, 'target': z, 'ratio': ratio, 'name': image_name}
        return sample


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    root_dir = "/home/uic55883/mapillary/training/"
    image_folder = "images_subset"
    mask_folder = "grayinstances_subset"
    train_dataset = Cardataset(root_dir, image_folder, mask_folder, 'train')
    # train_dataset.__getitem__(1)
    for i, sample in enumerate(train_dataset):
        data = sample['image']
        print(data.shape)
        # data=data.numpy()
        print(data.shape)
        # data=data.reshape(data.shape[1],data.shape[2],data.shape[0])
        data = data.permute(1, 2, 0)
        print(data.shape)
        plt.imshow(data.numpy())
        plt.show()
