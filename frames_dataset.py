import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2
import csv

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(video_dir):
  frames = []
  for im_path in os.listdir(video_dir):
    img = cv2.imread(im_path))[:, :, [2, 1, 0]]
    w, h, c = img.shape

    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1 + d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)

    img = (img/255.) * 2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)

    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)

    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def get_video_directories(root):
    video_directories = []
    for subdir in os.listdir(root):
        if os.path.isfile(subdir):
            continue
        video_directories.append(os.path.join(root, subdir))

    return dataset


class Frames(data_utl.Dataset):
    def __init__(self, root, mode, transforms=None, save_dir=''):
        self.video_directories = make_dataset(root)
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid_dir = self.video_directories[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(video_dir)
        else:
            imgs = load_flow_frames(self.root, vid, 1, nf)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), vid

    def __len__(self):
        return len(self.video_directories)
