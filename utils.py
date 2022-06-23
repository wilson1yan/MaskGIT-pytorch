import os
import glob
import random
import albumentations
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt

from dist_ops import get_size, get_rank


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class ImagePaths(data.Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = glob.glob(os.path.join(path, '*', '*.JPEG'))
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=256)
    sampler = data.distributed.DistributedSampler(
        train_data, num_replicas=get_size(), rank=get_rank()
    )
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size // get_size(),
                                   num_workers=4, pin_memory=True, sampler=sampler)
    return train_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images: dict):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    new_sample = images["new_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(new_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
