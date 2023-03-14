from collections import defaultdict
from copy import deepcopy
from prettytable import PrettyTable
import torch


class AvgResultProcessor:
    def __init__(self, label2name: dict):
        self.num_class = len(label2name.keys())
        self.label2name = deepcopy(label2name)
        self.acc_per_class = defaultdict(int)
        self.count_per_class = defaultdict(int)
        self.result_per_class = defaultdict(float)
        self.all_accurate = []
        for i in range(self.num_class):
            self.acc_per_class[i] = 0
            self.count_per_class[i] = 0

    def process(self, accurate, label):
        if isinstance(accurate, torch.Tensor):
            accurate = accurate.long().cpu()
            accurate = accurate.numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu()
            label = label.numpy()

        for acc, l in zip(accurate, label):
            self.count_per_class[l] += 1
            self.acc_per_class[l] += acc
            self.all_accurate.append(acc)

    def calculate(self):
        for c in range(self.num_class):
            self.result_per_class[c] = self.acc_per_class[c] / self.count_per_class[c] if self.count_per_class[c] > 0 else 0.0

    def info(self):
        t = PrettyTable(["Corruption", "Accuracy", "Error Rate"])
        for c in range(self.num_class):
            if c not in self.label2name.keys():
                continue
            t.add_row([self.label2name[c], f"{self.result_per_class[c] * 100:.2f}", f"{(1 - self.result_per_class[c]) * 100:.2f}"])
        # avg is the total avg
        avg = sum(self.all_accurate) / len(self.all_accurate)
        t.add_row(["Total Avg", f"{avg * 100:.2f}", f"{(1 - avg) * 100:.2f}"])

        per_domain_err = [f"{(1 - self.result_per_class[c]) * 100:.2f}" for c in range(self.num_class)]
        summary = '\t'.join(per_domain_err)
        info = f"{t} \n" \
               + "You should better calculate the per-class average by yourself!\n" \
               + f"summary: {summary}\n"
        return info

    def cumulative_acc(self):
        return sum(self.all_accurate) / len(self.all_accurate)
