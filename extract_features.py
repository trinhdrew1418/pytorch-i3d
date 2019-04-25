import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import code

import numpy as np

from pytorch_i3d import InceptionI3d

from frames_dataset import Frames


def run(max_steps=64e3, mode='rgb', root='../crossmodal_retrieval/data/YLI-MED-25rgb/', split='YLI_MED_folder_names.csv', batch_size=1, load_model='models/rgb_imagenet.pt', save_dir='./saved_feats'):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = Frames(root, mode, test_transforms, save_dir=save_dir)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    i3d.replace_logits(400)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs, name = data
        name = name[0].split('/')[-1]

        if os.path.exists(os.path.join(save_dir, name + '.npy')):
            continue

        b,c,t,h,w = inputs.shape
        limit = 250
        step = 200
        if t > limit:
            features = []
            for start in range(0, t-step, step):
                print(t)
                print(start)
                end = min(t, start+step)
                print(end)
                with torch.no_grad():
                    ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda())
                features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
            print(save_dir, name)
            np.save(os.path.join(save_dir, name), np.concatenate(features, axis=0))
        else:
            # wrap them in Variable
            with torch.no_grad():
                inputs = Variable(inputs.cuda())
            features = i3d.extract_features(inputs)
            print(save_dir, name[0])
            np.save(os.path.join(save_dir, name), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':
    # need to add argparse
    run()
