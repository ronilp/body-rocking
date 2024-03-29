# File: dataset_utils.py
# Author: Ronil Pancholia
# Date: 3/7/19
# Time: 2:51 AM

import os
import torch
import multiprocessing

import config


def load_datasets(DatasetClass, transforms=None, datasets=None):
    batch_size_dict = {}
    batch_size_dict['train'] = config.BATCH_SIZE
    batch_size_dict['val'] = 16 * config.BATCH_SIZE
    if datasets is None:
        datasets = {x: DatasetClass(os.path.join(config.DATA_DIR, x), x, transforms) for x in ['train', 'val']}
    dataset_loaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size_dict[x], shuffle=True,
                                       num_workers=multiprocessing.cpu_count()) for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    return dataset_loaders, dataset_sizes


def load_testset(DatasetClass, transforms=None, datasets=None, sessions=None):
    if datasets is None:
        datasets = {x: DatasetClass(os.path.join(config.DATA_DIR, x), x, transforms, sessions) for x in ['test']}
    dataset_loaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=config.BATCH_SIZE * 16, shuffle=False,
                                       num_workers=multiprocessing.cpu_count()) for x in ['test']}
    dataset_sizes = {x: len(datasets[x]) for x in ['test']}
    return dataset_loaders, dataset_sizes


def load_datasets_from_csv(DatasetClass, transforms=None):
    datasets = {x: DatasetClass(config.DATA_DIR, x, transforms) for x in ['train', 'val']}
    return load_datasets(DatasetClass, transforms, datasets)


def load_testset_from_csv(DatasetClass, transforms=None):
    datasets = {x: DatasetClass(config.DATA_DIR, x, transforms) for x in ['test']}
    return load_testset(DatasetClass, transforms, datasets)
