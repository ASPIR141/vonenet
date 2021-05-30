import os
import psutil
import time
import json
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class Collector(object):
    _stats     = dict()
    _reduce_dtype = torch.float32

    def report(self, name, value, mean=False):
        elems = torch.as_tensor(value)
        elems = elems.detach().flatten().to(self._reduce_dtype)

        if name in self._stats and mean:
            self._stats[name] = torch.mean(self._stats[name] + elems)
        else:
            self._stats[name] = elems

    def as_dict(self):
        stats = dict()
        for name, value in self._stats.items():
            stats[name] = float(value)
        return stats
    
    def flush(self):
        self._stats = { key: 0 for key in self._stats }


class Pipeline(object):
    def __init__(self, discriminator: nn.Module, generator: nn.Module,
                 g_optimizer: optim.Optimizer, d_optimizer: optim.Optimizer, adversarial_loss_fn, auxiliary_loss_fn,
                 dataset_len: int, num_classes: int, latent_dim: int, ngpus: int, device: torch.device, sample_interval=10, use_cuda=True,
                 prefix='', use_writer=True):
        self.discriminator = discriminator
        self.generator = generator
        self.adversarial_loss_fn = adversarial_loss_fn
        self.auxiliary_loss_fn = auxiliary_loss_fn
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.latent_dim = latent_dim
        self.prefix = prefix
        self.use_writer = use_writer
        self.dataset_len = dataset_len
        self.num_classes = num_classes
        self.sample_interval = sample_interval
        self.use_cuda = use_cuda & torch.cuda.is_available()
        self.device = device
        self.collector = Collector()

        if use_cuda and ngpus > 1 and torch.cuda.device_count() > 1:
            print('Running on multiple GPUs')
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)

        if use_cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
        else:
            print('Running on CPU')

    def get_samples(self, n_row):
        fixed_noise = torch.randn(n_row, self.latent_dim, device=self.device)
        labels = torch.randint(0, self.num_classes,
                               (n_row,), device=self.device)
        fake = self.generator(fixed_noise, labels)
        return fake

    def train(self, data_loader: DataLoader, epochs: int):
        start_time = time.time()
        os.makedirs(os.path.join('training-runs', self.prefix), exist_ok=True)
        stats_jsonl = open(os.path.join(f'training-runs/{self.prefix}', 'stats.jsonl'), 'wt') # Create run_dir parameter

        if self.use_writer:
            self.writer = SummaryWriter(
                f'training-runs/{self.prefix}', max_queue=100)

        # input = torch.rand(64, 3, 64, 64)
        # with torch.no_grad():
        #     self.writer.add_graph(self.discriminator, (input,))

        for epoch in tqdm.trange(0, epochs + 1, initial=0, desc='epoch'):
            for step, (data, target) in enumerate(tqdm.tqdm(data_loader)):

                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                batch_size = data.shape[0]
                labels = torch.full(
                    (batch_size, 1), 1., dtype=torch.float, device=self.device)

                # Train Generator
                self.g_optimizer.zero_grad()

                noise = torch.randn(
                    batch_size, self.latent_dim, device=self.device)
                gen_labels = torch.randint(
                    0, self.num_classes, (batch_size,), device=self.device)

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
                # labels.fill_(0.)
                fake_labels = torch.full(
                    (batch_size, 1), 0., dtype=torch.float, device=self.device)

                fake_pred, fake_aux = self.discriminator(fake.detach())
                d_fake_loss = (self.adversarial_loss_fn(
                    fake_pred, fake_labels) + self.auxiliary_loss_fn(fake_aux, gen_labels)) / 2

                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward(retain_graph=True)
                self.d_optimizer.step()

                # Calculate discriminator accuracy
                pred = torch.cat(
                    [real_aux.data.cpu(), fake_aux.data.cpu()], axis=0)
                gt = torch.cat(
                    [target.data.cpu(), gen_labels.data.cpu()], axis=0)
                d_acc = torch.mean(torch.Tensor.float(torch.argmax(pred, axis=1) == gt))

                tick_end_time = time.time()
                self.collector.report('Train/d_loss', d_loss, mean=True)
                self.collector.report('Train/g_loss', g_loss, mean=True)
                self.collector.report('Train/accuracy', 100 * d_acc, mean=True)
                self.collector.report('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
                self.collector.report('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30)
                self.collector.report('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(self.device) / 2**30)

                batches_done = epoch * len(data_loader) + step
                if batches_done % self.sample_interval == 0:
                    stats_dict = self.collector.as_dict()

                    if stats_jsonl is not None:
                        stats_jsonl.write(json.dumps(stats_dict) + '\n')
                        stats_jsonl.flush()

                    if self.writer is not None:
                        for name, value in stats_dict.items():
                            self.writer.add_scalar(name, value / self.sample_interval, batches_done)
                        self.writer.add_image('Train/samples', make_grid(self.get_samples(20), normalize=True), batches_done)
                        self.writer.flush()

                    self.collector.flush()
