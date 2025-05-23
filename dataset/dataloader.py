import os
import pickle
import numpy as np

from collections import OrderedDict
from torch.utils.data import Dataset


class Dataset3D_itemread(Dataset):
    def __init__(self, data, patch_size):
        super(Dataset3D_itemread, self).__init__()
        self._data = data
        self.patch_size = patch_size
        print(self.patch_size)

        sels = list(self._data.keys())
        self.datas, self.datas_defs, self.deaths, self.ostimes, self.infos, self.subtypes, self.recurrences, self.names, self.shapes = [], [], [], [], [], [], [], [], []
        for name in sels:
            data = np.load(self._data[name]['path'])['data']
            data_def = np.load(self._data[name]['path1'])['data']
            death = np.load(self._data[name]['path'])['data_death']
            age = np.load(self._data[name]['path'])['data_age']
            tumorpos = np.load(self._data[name]['path'])['data_tumorpos']
            if death == 2:
                death = 0
            else:
                death = int(death)

            ostime = np.load(self._data[name]['path'])['data_ostime']
            ostime = ostime.astype(np.float32)
            info = np.concatenate(([age], tumorpos))
            info = info.astype(np.float32)

            shape = np.array(data.shape[1:])
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]),
                                 (pad_left[2], pad_right[2])))
            shape = np.array(data_def.shape)
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            data_def = np.pad(data_def, ((pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]),
                                         (pad_left[2], pad_right[2])))

            data_def = (data_def - data_def.mean()) / (data_def.std() + 1e-8)

            images = data[:-1]
            self.datas.append(images)
            self.datas_defs.append(data_def)
            self.deaths.append(death)
            self.ostimes.append(ostime)
            self.infos.append(info)
            self.names.append(name)
            self.shapes.append(shape)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return {'data': self.datas[index], 'data_def': self.datas_defs[index], 'death': self.deaths[index], 'ostime': self.ostimes[index], 'info': self.infos[index], 'name': self.names[index], 'shape': self.shapes[index]}


class Dataset3D_itemread_BraTS20(Dataset):
    def __init__(self, data, patch_size):
        super(Dataset3D_itemread_BraTS20, self).__init__()
        self._data = data
        self.patch_size = patch_size
        print(self.patch_size)

        sels = list(self._data.keys())
        self.datas, self.datas_defs, self.deaths, self.ostimes, self.infos, self.subtypes, self.recurrences, self.names, self.shapes = [], [], [], [], [], [], [], [], []
        for name in sels:
            data = np.load(self._data[name]['path'])['data']
            data_def = np.load(self._data[name]['path1'])['data']
            death = np.load(self._data[name]['path'])['data1']
            age = np.load(self._data[name]['path'])['data_age']
            tumorpos = np.load(self._data[name]['path'])['data_tumorpos']
            if death == 2:
                death = 0
            else:
                death = int(death)

            ostime = np.load(self._data[name]['path'])['data_ostime']
            ostime = ostime.astype(np.float32)
            info = np.concatenate(([age], tumorpos))
            info = info.astype(np.float32)

            shape = np.array(data.shape[1:])
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]),
                                 (pad_left[2], pad_right[2])))
            shape = np.array(data_def.shape)
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            data_def = np.pad(data_def, ((pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]),
                                         (pad_left[2], pad_right[2])))

            data_def = (data_def - data_def.mean()) / (data_def.std() + 1e-8)

            images = data[:-1]
            self.datas.append(images)
            self.datas_defs.append(data_def)
            self.deaths.append(death)
            self.ostimes.append(ostime)
            self.infos.append(info)
            self.names.append(name)
            self.shapes.append(shape)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return {'data': self.datas[index], 'data_def': self.datas_defs[index], 'death': self.deaths[index], 'ostime': self.ostimes[index], 'info': self.infos[index], 'name': self.names[index], 'shape': self.shapes[index]}


def get_dataset(fold, split_file, mode, config, root='default', def_root='default'):
    if root == 'default':
        root = config.DATASET.ROOT
        def_root = config.DATASET.DEF_ROOT
    with open(os.path.join(root, split_file), 'rb') as f:
        if fold == -1:
            splits = pickle.load(f)
        else:
            splits = pickle.load(f)[fold]
    if mode == 'train':
        datas = splits['train']
    elif mode == 'val':
        datas = splits['val']
    else:
        datas = splits['test']
    print(datas.shape)
    dataset = OrderedDict()
    for name in datas:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(root, name + '.npz')
        dataset[name]['path1'] = os.path.join(def_root, name + '.npz')
    assert len(config.MODEL.INPUT_SIZE) == 3, 'must be 3 dimensional patch size'
    return Dataset3D_itemread(dataset, config.MODEL.INPUT_SIZE)


def get_dataset_BraTS20(mode, config):
    with open(os.path.join(config.DATASET.TEST_ROOT, 'splits_2020.pkl'), 'rb') as f:
        splits = pickle.load(f)
    datas = splits['train'] if mode == 'train' else splits['val']
    print(datas.shape)
    dataset = OrderedDict()
    for name in datas:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(config.DATASET.TEST_ROOT, name + '.npz')
        dataset[name]['path1'] = os.path.join(config.DATASET.TEST_DEF_ROOT, name + '.npz')
    assert len(config.MODEL.INPUT_SIZE) == 3, 'must be 3 dimensional patch size'
    return Dataset3D_itemread_BraTS20(dataset, config.MODEL.INPUT_SIZE)
