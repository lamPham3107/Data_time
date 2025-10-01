# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
# FIX: Thêm try-except cho import
try:
    from data.dataset import SetDataset_JSON, SimpleDataset, SetDataset, EpisodicBatchSampler, SimpleDataset_JSON
except ImportError:
    from .dataset import SetDataset_JSON, SimpleDataset, SetDataset, EpisodicBatchSampler, SimpleDataset_JSON
from abc import abstractmethod
import os



class TransformLoader:
    def __init__(self, image_size):
        self.normalize_param = dict(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        
        self.image_size = image_size
        if image_size == 112:
            self.resize_size = 112
        elif image_size == 224:
            self.resize_size = 256

    def get_composed_transform(self, aug=False):
        if aug:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize_param)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize_param)
            ])
        return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


# Helper để tạo DataLoader; trên Windows đặt num_workers=0 để tránh lỗi pickle với lambda/closure
def make_dataloader(dataset, batch_size=None, shuffle=False, num_workers=None, pin_memory=True, batch_sampler=None):
    if num_workers is None:
        if os.name == 'nt':  # Windows
            num_workers = 0
        else:
            try:
                import multiprocessing
                num_workers = min(12, max(0, multiprocessing.cpu_count() - 1))
            except Exception:
                num_workers = 4
    params = {}
    if batch_sampler is not None:
        params['batch_sampler'] = batch_sampler
    else:
        params['batch_size'] = batch_size
        params['shuffle'] = shuffle
    params['num_workers'] = num_workers
    params['pin_memory'] = pin_memory
    return torch.utils.data.DataLoader(dataset, **params)


class SimpleDataManager(DataManager):
    def __init__(self, data_path, image_size, batch_size, json_read=False):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.trans_loader = TransformLoader(image_size)
        self.json_read = json_read

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        if self.json_read:
            dataset = SimpleDataset_JSON(self.data_path, data_file, transform)
        else:
            dataset = SimpleDataset(self.data_path, data_file, transform)
        # Dùng helper make_dataloader; num_workers mặc định sẽ về 0 nếu chạy trên Windows
        data_loader = make_dataloader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, data_path, image_size, n_way, n_support, n_query, n_episode, json_read=False):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.data_path = data_path
        self.json_read = json_read

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        if self.json_read:
            dataset = SetDataset_JSON(self.image_size, self.data_path, data_file, self.batch_size, transform)
        else:
            dataset = SetDataset(self.data_path, data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        # Sử dụng batch_sampler (không truyền batch_size/shuffle)
        data_loader = make_dataloader(dataset, batch_sampler=sampler, pin_memory=True)
        return data_loader



