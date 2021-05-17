import time

import torch
import torch.nn as nn
from torch.optim import Adam

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class DCGANTrainer(object):

    def __init__(self, discriminator: nn.Module, generator: nn.Module, device: torch.device,
                 lr: float, b1: float, img_size: int, batch_size: int, num_workers: int, latent_dim: int):
        self.name = 'dcgan_train'
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.criterion = nn.BCELoss()
        self.g_optimizer = Adam(self.generator.parameters(),
                                lr=lr, betas=(b1, 0.999))
        self.d_optimizer = Adam(self.discriminator.parameters(),
                                lr=lr, betas=(b1, 0.999))
        self.real_label = 1.
        self.fake_label = 0.
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.latent_dim = latent_dim

        self.data_loader = self.data()

    def data(self):
        dataset = datasets.MNIST(
            './data',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]))

        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return data_loader

    def get_sample(self):
        fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
        fake = self.generator(fixed_noise)
        return fake

    def __call__(self, inp, target):
        start = time.time()

        self.discriminator.zero_grad()

        # Train Discriminator
        real_input = inp.to(self.device)
        b_size = real_input.size(0)
        label = torch.full((b_size,), self.real_label,
                           dtype=torch.float, device=self.device)

        output = self.discriminator(real_input).view(-1)
        d_real_loss = self.criterion(output, label)

        d_real_loss.backward()

        noise = torch.randn(b_size, self.latent_dim, 1, 1, device=self.device)
        fake = self.generator(noise)

        label.fill_(self.fake_label)

        output = self.discriminator(fake.detach()).view(-1)
        d_fake_loss = self.criterion(output, label)

        d_loss = d_real_loss + d_fake_loss

        d_fake_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.generator.zero_grad()
        label.fill_(self.real_label)

        output = self.discriminator(fake).view(-1)
        g_loss = self.criterion(output, label)

        g_loss.backward()
        self.g_optimizer.step()

        record = {}
        record['d_loss'] = d_loss.item()
        record['g_loss'] = g_loss.item()
        record['duration'] = time.time() - start

        return record
