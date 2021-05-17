import os
import argparse
import subprocess
import io
import shlex
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from pipeline import Pipeline
from vone.vonenet import VOneNet
from networks.acgan import Discriminator, Generator
from networks import weights_init

import pandas as pd

parser = argparse.ArgumentParser(description='training')


parser.add_argument('--n_epochs', default=100, type=int,
                    help='number of epochs of training')
parser.add_argument('--batch_size', default=128, type=int,
                    help='size of batch for training')
parser.add_argument('--workers', default=2,
                    help='number of data loading workers')
parser.add_argument('--ngpus', default=2, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('--img_size', default=64, type=int,
                    help='size of each image dimension')
parser.add_argument('--channels', default=3, type=int,
                    help='number of image channels')
parser.add_argument('--latent_dim', default=100, type=int,
                    help='dimensionality of the latent space')
parser.add_argument('--n_classes', default=10, type=int,
                    help='number of classes for dataset')
parser.add_argument('--lr', default=0.0002, type=float,
                    help='adam: learning rate')
parser.add_argument('--b1', default=0.5, type=float,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', default=0.999, type=float,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--sample_interval', type=int, default=1000,
                    help='interval between image sampling')
parser.add_argument('--model_arch', choices=['acgan', 'dcgan'], default='dcgan',
                    help='back-end model architecture to load')

FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=2):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    if n > 0:
        gpus = subprocess.run(shlex.split(
            'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
            stdout=subprocess.PIPE).stdout
        gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
        gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            visible = [int(i)
                       for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            gpus = gpus[gpus['index'].isin(visible)]
        gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
        # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


set_gpus(FLAGS.ngpus)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if is_available() else 'cpu')


def get_data_loader():
    dataset = datasets.CIFAR10(
        './data/CIFAR10',
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.Resize(FLAGS.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]))

    data_loader = DataLoader(dataset,
                             batch_size=FLAGS.batch_size,
                             shuffle=True,
                             num_workers=FLAGS.workers,
                             pin_memory=True)

    return data_loader


def main():
    generator = Generator(img_size=FLAGS.img_size, n_classes=FLAGS.n_classes,
                          channels=FLAGS.channels, latent_dim=FLAGS.latent_dim)
    discriminator = Discriminator(
        img_size=FLAGS.img_size, n_classes=FLAGS.n_classes, channels=FLAGS.channels)

    discriminator.apply(weights_init)
    generator.apply(weights_init)

    v_one = VOneNet(image_size=FLAGS.img_size, stride=2,
                    ksize=3, in_channels=FLAGS.channels)
    bottleneck = nn.Conv2d(512, 16, kernel_size=1, stride=1, bias=False)
    nn.init.kaiming_normal_(
        bottleneck.weight, mode='fan_out', nonlinearity='relu')

    discriminator = nn.Sequential(OrderedDict([
        ('vone_block', v_one),
        ('bottleneck', bottleneck),
        ('model', discriminator),
    ]))

    g_optimizer = Adam(generator.parameters(), lr=FLAGS.lr,
                       betas=(FLAGS.b1, FLAGS.b2))
    d_optimizer = Adam(discriminator.parameters(),
                       lr=FLAGS.lr, betas=(FLAGS.b1, FLAGS.b2))

    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()

    data_loader = get_data_loader()

    pipeline = Pipeline(discriminator, generator, g_optimizer, d_optimizer, adversarial_loss,
                        auxiliary_loss, len(data_loader), FLAGS.n_classes, prefix=f'vone_{FLAGS.model_arch}')
    pipeline.train(data_loader, FLAGS.epochs)


if __name__ == '__main__':
    main()
