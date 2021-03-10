#!/usr/bin/env python
import os
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pandas as pdun
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict


class Epoch:
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None
        self.duration = None


class Run:
    def __init__(self):
        self.params = None
        self.count = 0
        self.data = []
        self.start_time = None
        self.duration = None


class RunCycleManager:
    """ [ Insert documentation ] """

    def __init__(self) -> None:
        """ [ Insert documentation ] """
        self.run = Run()
        self.epoch = Epoch()
        self.netG_A2B = None
        self.netG_B2A = None
        self.netD_A = None
        self.netD_B = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, device, netG_A2B, netG_B2A, netD_A, netD_B, loader) -> None:
        """ [ Insert documentation ] """
        self.run.start_time = time.time()
        self.run.params = run
        self.run.count += 1

        self.netG_A2B = netG_A2B
        self.netG_B2A = netG_B2A
        self.netD_A = netD_A
        self.netD_B = netD_B

        self.loader = loader
        self.tb = SummaryWriter(comment=f"-{run}")

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image("images", grid)
        self.tb.add_graph(self.netG_A2B, images.to(device))
        self.tb.add_graph(self.netG_B2A, images.to(device))
        self.tb.add_graph(self.netD_A, images.to(device))
        self.tb.add_graph(self.netD_B, images.to(device))

    def end_run(self) -> None:
        """ [ Insert documentation ] """
        self.tb.close()
        self.epoch.count = 0

    def begin_epoch(self) -> None:
        """ [ Insert documentation ] """
        self.epoch.start_time = time.time()
        self.epoch.count += 1
        self.epoch.loss = 0
        self.epoch.num_correct = 0

    def end_epoch(self, save_runs=True, print_df=False) -> None:
        """ [ Insert documentation ] """
        self.epoch.duration = time.time() - self.epoch.start_time
        self.run.duration = time.time() - self.run.start_time

        loss = self.epoch.loss / len(self.loader.dataset)
        accuracy = self.epoch.num_correct / len(self.loader.dataset)

        self.tb.add_scalar("Loss", loss, self.epoch.count)
        self.tb.add_scalar("Accuracy", accuracy, self.epoch.count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch.count)
            self.tb.add_histogram(f"{name}.grad", param.grad, self.epoch.count)

        results = OrderedDict()
        results["run"] = self.run.count
        results["epoch"] = self.epoch.count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = self.epoch.duration
        results["run duration"] = self.run.duration
        for k, v in self.run.params._asdict().items():
            results[k] = v
        self.run.data.append(results)

        if print_df:
            os.system("cls")
            pprint.pprint(
                pd.DataFrame.from_dict(self.run.data, orient="columns").sort_values(
                    "accuracy", ascending=False
                )
            )

    def track_loss(self, loss, batch) -> None:
        """ [ Insert documentation ] """
        self.epoch.loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels) -> None:
        """ [ Insert documentation ] """
        self.epoch.num_correct += self.get_num_correct(preds, labels)

    def get_num_correct(self, preds, labels) -> int:
        """ [ Insert documentation ] """
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName) -> None:
        """ [ Insert documentation ] """

        pd.DataFrame.from_dict(self.run.data, orient="columns").to_csv(
            f"{fileName}.csv"
        )

        with open(f"{fileName}.json", "w", encoding="utf-8") as f:
            json.dump(self.run.data, f, ensure_ascii=False, indent=4)

        # print(f"Saved results to: {fileName}.json && {fileName}.csv")

