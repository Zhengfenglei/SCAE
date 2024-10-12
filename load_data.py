import gc
import h5py
import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import copy
from DataAugment import *

class LDED1_30(data.Dataset):
    def __init__(self, dataset, label):
        self.data = dataset
        self.label = label

    def __getitem__(self, index):
        clip = self.data[index]
        numpy_clip = clip.numpy()
        # L2norm
        l2norm = np.linalg.norm(numpy_clip)
        clip = clip / l2norm

        # max_min_norm
        # min_val = numpy_clip.min()
        # max_val = numpy_clip.max()
        # clip = (clip - min_val) / (max_val - min_val)

        # L1
        # l1norm = torch.abs(clip).sum()
        # clip = clip / l1norm

        # z-score
        # mean = numpy_clip.mean()
        # std = numpy_clip.std()
        # clip = (clip - mean) / std

        label = self.label[index]

        return clip, label

    def __len__(self):
        return len(self.data)

def numpy_To_PIL(numpy):
    numpy = numpy.astype(np.uint8)
    image = Image.fromarray(numpy)

    return image

class AugDataset(data.Dataset):
    def __init__(self, data, targets, transform):
        super(AugDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        clip = [img[:, :, frame] for frame in range(150)]
        self.transform.randomize_parameters()
        imgw = [self.transform(numpy_To_PIL(img_)) for img_ in clip]
        imgw = torch.cat(imgw, 0)

        return img, imgw, target

    def __len__(self):
        return len(self.data)

#  train the autoencoder
def load_data(BATCH_SIZE):
    split_ratio = 0.8
    BASE_DIR = '/'
    INPUT_DIR = 'input.mat'
    OUTPUT_DIR = 'label.mat'
    INPUT_DIR = os.path.join(BASE_DIR, INPUT_DIR)
    OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR)

    label = h5py.File(name=OUTPUT_DIR, mode='r')
    label = label['label'][()]
    label = label.squeeze()

    indices = list(range(len(label)))
    random.shuffle(indices)

    label = label[indices]
    label_train = label[:int(label.shape[0] * split_ratio)]
    label_test = label[int(label.shape[0] * split_ratio):]
    label_train = torch.from_numpy(label_train.squeeze()).type(torch.LongTensor)
    label_test = torch.from_numpy(label_test.squeeze()).type(torch.LongTensor)
    del label
    gc.collect()
    import scipy.io as scio

    data = scio.loadmat(INPUT_DIR)
    input = data['data']
    input = np.transpose(input, (3, 2, 1, 0))
    input = input[indices][:][:][:]
    input_train = input[:int(input.shape[0] * split_ratio)][:][:][:]
    input_test = input[int(input.shape[0] * split_ratio):][:][:][:]
    input_train = torch.from_numpy(input_train).type(torch.LongTensor) / 255.0
    input_test = torch.from_numpy(input_test).type(torch.LongTensor) / 255.0
    del input
    gc.collect()

    train_data = LDED1_30(input_train, label_train)
    test_data = LDED1_30(input_test, label_test)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        drop_last=True)
    return train_loader, test_loader

#  train the classification model
def load_semi_data(BATCH_SIZE, label_ratio = 0.2):
    split_ratio = 0.8
    BASE_DIR = '/'
    INPUT_DIR = 'input.mat'
    OUTPUT_DIR = 'label.mat'
    INPUT_DIR = os.path.join(BASE_DIR, INPUT_DIR)
    OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR)

    label = h5py.File(name=OUTPUT_DIR, mode='r')
    label = label['label'][()]
    label = label.squeeze()

    indices = list(range(len(label)))
    random.shuffle(indices)

    label = label[indices]
    label_train = label[:int(label.shape[0] * split_ratio * label_ratio)]
    label_test = label[int(label.shape[0] * split_ratio):]

    label_train = torch.from_numpy(label_train.squeeze()).type(torch.LongTensor)
    label_test = torch.from_numpy(label_test.squeeze()).type(torch.LongTensor)
    del label
    gc.collect()
    import scipy.io as scio

    data = scio.loadmat(INPUT_DIR)
    input = data['data']
    input = np.transpose(input, (3, 2, 1, 0))
    input = input[indices][:][:][:]
    input_train = input[: int(input.shape[0] * split_ratio * label_ratio)][:][:][:]
    input_test = input[int(input.shape[0] * split_ratio):][:][:][:]
    input_train = torch.from_numpy(input_train).type(torch.LongTensor) / 255.0
    input_test = torch.from_numpy(input_test).type(torch.LongTensor) / 255.0
    gc.collect()

    train_data = LDED1_30(input_train, label_train)
    test_data = LDED1_30(input_test, label_test)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        drop_last=True)
    return train_loader, test_loader

def load_aug_data(BATCH_SIZE):
    split_ratio = 0.8
    BASE_DIR = '/'
    INPUT_DIR = 'input.mat'
    OUTPUT_DIR = 'label.mat'
    INPUT_DIR = os.path.join(BASE_DIR, INPUT_DIR)
    OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR)

    label = h5py.File(name=OUTPUT_DIR, mode='r')
    label = label['label'][()]
    label = label.squeeze()

    indices = list(range(len(label)))
    random.shuffle(indices)

    label = label[indices]
    label_train = label[:int(label.shape[0] * split_ratio)]
    label_test = label[int(label.shape[0] * split_ratio):]
    label_train = torch.from_numpy(label_train.squeeze()).type(torch.LongTensor)
    label_test = torch.from_numpy(label_test.squeeze()).type(torch.LongTensor)
    del label
    gc.collect()
    import scipy.io as scio

    data = scio.loadmat(INPUT_DIR)
    input = data['data']
    input = np.transpose(input, (3, 1, 0, 2))  # (32, 32, 150, 2286) -->(2286, 32, 32, 150) (3, 1, 0, 2)
    input = input[indices][:][:][:]
    input_train = input[:int(input.shape[0] * split_ratio)][:][:][:]
    input_test = input[int(input.shape[0] * split_ratio):][:][:][:]
    del input
    gc.collect()

    crop_method = MultiScaleCornerCrop([1, 0.7, 0.9, 0.85, 0.8], 32, crop_positions='c')
    train_transform = Compose([
        RandomHorizontalFlip(),
        # RandomRotate(),
        # RandomResize(),
        crop_method,
        # MultiplyValues(),
        # Dropout(),
        # SaltImage(),
        # Gaussian_blur(),
        # SpatialElasticDisplacement(),
        ToTensor(255),
        # Normalize([0, 0, 0], [1, 1, 1])
    ])

    test_transform = Compose([
        ToTensor(255),
    ])
    train_data = AugDataset(input_train, label_train, train_transform)
    test_data = AugDataset(input_test, label_test, test_transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        drop_last=True)
    return train_loader, test_loader

