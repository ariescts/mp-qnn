import os
import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


_LOCAL_DATA_PATH = './datasets'
_IMAGENET_MAIN_PATH = '/home/datasets/ILSVRC2012'
_DAMAGENET_PATH = '/home/datasets/DAmageNet/DAmageNet'


_data_path = {
    'cifar10': os.path.join(_LOCAL_DATA_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_LOCAL_DATA_PATH, 'CIFAR100'),
    'stl10': os.path.join(_LOCAL_DATA_PATH, 'STL10'),
    'mnist': os.path.join(_LOCAL_DATA_PATH, 'MNIST'),
    'damagenet': _DAMAGENET_PATH,
    'imagenet': {
        'train': os.path.join(_IMAGENET_MAIN_PATH, 'train'),
        'val': os.path.join(_IMAGENET_MAIN_PATH, 'val'),
    },
}


class SingleDataset(datasets.ImageFolder):
    def __init__(self, root, transform, target_transform):
        super(SingleDataset, self).__init__(self, root=root, transform=transform,
                                            target_transform=target_transform)

    def get_item(self, idx):
        return self.__getitem__(self, index=idx)


def get_dataset(name, split='train', transform=None, target_transform=None, download=True):
    train = split == 'train'
    path = _data_path[name]
    if name == 'cifar10':
        return datasets.CIFAR10(root=path, train=train, transform=transform,
                                target_transform=target_transform, download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=path, train=train, transform=transform,
                                 target_transform=target_transform, download=download)
    elif name == 'damagenet':
        path = _DAMAGENET_PATH
        return datasets.ImageFolder(root=path, transform=transform,
                                    target_transform=target_transform)
    elif name == 'imagenet':
        path = path[split]
        return datasets.ImageFolder(root=path, transform=transform,
                                    target_transform=target_transform)
    elif name == 'single_damagenet':
        path = _DAMAGENET_PATH
        return SingleDataset(root=path, transform=transform)
    elif name == 'single_imagenet':
        path = path[split]
        return datasets.ImageFolder(root=path, transform=transform,
                                    target_transform=target_transform)
    else:
        raise ValueError('Unsupported dataset name!')


def create_dataloder_cifar(name, transform, vr=0, batch_size=128, num_workers=0):
    dataset_train = get_dataset(name, split='train', transform=transform['train'])
    dataset_val = get_dataset(name, split='train', transform=transform['eval'])
    dataset_test = get_dataset(name, split='test', transform=transform['eval'])

    idx_sorted = np.argsort(dataset_train.targets)
    num_classes = dataset_train.targets[idx_sorted[-1]] + 1
    samples_per_class = len(dataset_train) // num_classes
    val_len = int(vr * samples_per_class)
    val_idx = np.array([], dtype=np.int32)
    train_idx = np.array([], dtype=np.int32)
    for i in range(num_classes):
        perm = np.random.permutation(range(samples_per_class))

        val_part = samples_per_class * i + perm[0:val_len]
        val_part = idx_sorted[val_part]
        val_idx = np.concatenate((val_idx, val_part))

        train_part = samples_per_class * i + perm[val_len:]
        train_part = idx_sorted[train_part]
        train_idx = np.concatenate((train_idx, train_part))

    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               shuffle=False, sampler=sampler_train,
                                               num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                             shuffle=False, sampler=sampler_val,
                                             num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers,
                                              pin_memory=True)

    return train_loader, val_loader, test_loader


def create_dataloader_imagenet(name, transform, batch_size, workers):
    dataset_train = get_dataset(name, split='train', transform=transform['train'])
    dataset_val = get_dataset(name, split='val', transform=transform['eval'])

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               shuffle=True, num_workers=workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size//2,
                                              shuffle=False, num_workers=workers,
                                              pin_memory=True)

    return train_loader, val_loader


def create_dataloader_damagenet(name, transform, batch_size, workers):
    dataset = get_dataset(name, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers,
                                             pin_memory=True)

    return dataloader



