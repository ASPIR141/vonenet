import os
import psutil
import time
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class Pipeline(object):
    def __init__(self, discriminator: nn.Module, generator: nn.Module,
                 g_optimizer: optim.Optimizer, d_optimizer: optim.Optimizer, adversarial_loss_fn, auxiliary_loss_fn,
                 dataset_len: int, num_classes: int, sample_interval=10, use_cuda=True, prefix='', use_writer=True):
        self.discriminator = discriminator
        self.generator = generator
        self.adversarial_loss_fn = adversarial_loss_fn
        self.auxiliary_loss_fn = auxiliary_loss_fn
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.prefix = prefix
        self.use_writer = use_writer
        self.dataset_len = dataset_len
        self.num_classes = num_classes
        self.sample_interval = sample_interval
        self.use_cuda = use_cuda & torch.cuda.is_available()

        if use_cuda and torch.cuda.device_count() > 1:
            print('Running on multiple GPUs')
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)

        if use_cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
        else:
            print('Running on CPU')

    def get_sample(self, n_row):
        fixed_noise = torch.randn(
            n_row, self.latent_dim, 1, 1, device=self.device)
        labels = torch.randint(0, self.num_classes, n_row, device=self.device)
        fake = self.generator(fixed_noise, labels)
        return fake

    def train(self, data_loader: DataLoader, epochs: int):
        nsteps = len(data_loader)

        stats_dict = {}
        stats_dict['d_loss'] = 0.0
        stats_dict['g_loss'] = 0.0
        stats_dict['accuracy'] = 0.0

        if self.use_writer:
            self.writer = SummaryWriter(f'training-runs/{self.prefix}', max_queue=100)

        for epoch in tqdm.trange(0, epochs + 1, initial=0, desc='epoch'):
            for step, (data, target) in enumerate(tqdm.tqdm(data_loader)):
                start = time.time()

                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                batch_size = data.shape[0]
                labels = torch.full((batch_size,), 1.,
                                    dtype=torch.float, device=self.device)

                # Train Generator
                self.g_optimizer.zero_grad()

                noise = torch.randn(
                    batch_size, self.latent_dim, 1, 1, device=self.device)
                gen_labels = torch.randint(
                    0, self.num_classes, batch_size, device=self.device)

                fake = self.generator(noise, gen_labels)
                validity, pred_label = self.discriminator(fake)

                g_loss = 0.5 * (self.adversarial_loss_fn(validity, labels) +
                                self.auxiliary_loss_fn(pred_label, gen_labels))

                g_loss.backward()
                self.g_optimizer.step()

                # Train Discriminator
                self.d_optimizer.zero_grad()

                # Loss for real images
                real_pred, real_aux = self.discriminator(data)
                d_real_loss = (self.adversarial_loss_fn(
                    real_pred, labels) + self.auxiliary_loss_fn(real_aux, target)) / 2

                # Loss for fake images
                labels.fill_(0.)

                fake_pred, fake_aux = self.discriminator(fake.detach())
                d_fake_loss = (self.adversarial_loss_fn(
                    fake_pred, labels) + self.auxiliary_loss_fn(fake_aux, gen_labels)) / 2

                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.d_optimizer.step()

                # Calculate discriminator accuracy
                pred = torch.cat(
                    [real_aux.data.cpu(), fake_aux.data.cpu()], axis=0)
                gt = torch.cat(
                    [target.data.cpu(), gen_labels.data.cpu()], axis=0)
                d_acc = torch.mean(torch.argmax(pred, axis=1) == gt)

                stats_dict['Train/d_loss'] += d_loss.item()
                stats_dict['Train/g_loss'] += g_loss.item()
                stats_dict['Train/accuracy'] += 100 * d_acc
                stats_dict['Train/duration'] = time.time() - start
                stats_dict['Resources/cpu_mem_gb'] = psutil.Process(os.getpid()).memory_info().rss / 2**30
                stats_dict['Resources/peak_gpu_mem_gb'] = torch.cuda.max_memory_allocated(self.device) / 2**30
                
                fields = []
                fields += [f"cpumem {stats_dict['Resources/cpu_mem_gb']:<6.2f}"]
                fields += [f"gpumem {stats_dict['Resources/peak_gpu_mem_gb']:<6.2f}"]
                print(' '.join(fields))

                batches_done = epoch * nsteps + step
                if batches_done % self.sample_interval == 0:
                    for name, value in stats_dict.items():
                        self.writer.add_scalar(
                            name, value / self.sample_interval, batches_done)
                    self.writer.add_image(
                        'Train/samples', make_grid(self.get_sample(), normalize=True), batches_done)

                    stats_dict.clear()
