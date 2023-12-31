import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transforms import get_test_transforms, get_train_transforms, get_s10_train_transforms
import numpy as np
import pytorch_lightning as pl

dataset_mean, dataset_std = (0.4914, 0.4822, 0.4465), \
            (0.2470, 0.2435, 0.2616)

def get_dataset_mean_variance(dataset):

    if dataset_mean and dataset_std:
        return dataset_std, dataset_std

    imgs = [item[0] for item in dataset]
    imgs = torch.stack(imgs, dim=0)

    mean = []
    std = []
    for i in range(imgs.shape[1]):
        mean.append(imgs[:, i, :, :].mean().item())
        std.append(imgs[:, i, :, :].std().item())

    return tuple(mean), tuple(std)


def get_loader(**kwargs):

    dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.ToTensor())
    mean, std = get_dataset_mean_variance(dataset)
    if kwargs.get('s10'):
        train_data = CustomCIFAR10Dataset(train=True, transform=get_s10_train_transforms(mean, std, 0.3))
        del kwargs['s10']
    else:
        train_data = CustomCIFAR10Dataset(train=True, transform=get_train_transforms(mean, std, 0.5))
    test_data = CustomCIFAR10Dataset(train=False, transform=get_test_transforms(mean=mean, std=std))

    return torch.utils.data.DataLoader(train_data, **kwargs), torch.utils.data.DataLoader(test_data, **kwargs)


def get_dataset_labels():
    return ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def get_data_label_name(idx):
    if idx < 0:
        return ''

    return get_dataset_labels()[idx]


def get_data_idx_from_name(name):
    if not name:
        return -1

    return get_dataset_labels.index(name.lower()) if name.lower() in get_dataset_labels() else -1



class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root_dir='../data', train=True, download=True, transform=None):
        self.transform = transform
        self.dataset = datasets.CIFAR10(root_dir, train=train, download=download)
        self.root_dir = root_dir


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        data = self.dataset[index][0]
        target = self.dataset[index][1]

        #img = np.array(data)

        if self.transform:
            for tf in self.transform:
                if tf.get('type') == 't':
                    data = tf.get('value')(data)
                elif tf.get('type') == 'a':
                    img = np.array(data)
                    augmentations = tf.get('value')(image=img)
                    img = augmentations["image"]
                    data = torch.from_numpy(img.transpose(2, 0, 1))


        target = torch.from_numpy(np.array(target))


        return data, target


'''
Assignment - S12
'''

import albumentations as A
from torchvision import transforms as T
class CustomCIFARR10LightningDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super(CustomCIFARR10LightningDataModule, self).__init__()
        self.kwargs = kwargs

    def prepare_data(self):

        CustomCIFAR10Dataset('../data', train=True, download=True)
        CustomCIFAR10Dataset('../data', train=False, download=True)
        dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.ToTensor())
        self.mean, self.std = get_dataset_mean_variance(dataset)

    def get_train_transforms(self, p):
        t = T.Compose(
            [
                T.RandomCrop((32, 32), padding=4, fill=(self.mean[0] * 255, self.mean[1] * 255, self.mean[2] * 255))
            ]
        )

        a = A.Compose(
            [
                A.Normalize(self.mean, self.std),
                A.HorizontalFlip(p=p),
                A.CoarseDropout(max_holes=1,
                                max_height=16,
                                max_width=16,
                                min_holes=1,
                                min_height=16,
                                min_width=16,
                                fill_value=self.mean,
                                mask_fill_value=None,
                                p=p
                                )
            ]
        )

        #
        return [dict(type='t', value=t), dict(type='a', value=a)]


    def get_test_transforms(self):
        # Test data transformations
        a = A.Compose([
            A.Normalize(self.mean, self.std)
        ])

        return [dict(type='a', value=a)]


    def setup(self, stage):
        self.train_data = CustomCIFAR10Dataset('../root', train=True, download=False,
                                          transform=self.get_train_transforms(0.5))

        self.test_data = CustomCIFAR10Dataset('../root', train=False, download=False, transform=self.get_test_transforms())


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, **self.kwargs)


    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, **self.kwargs)

