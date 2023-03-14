from copy import deepcopy
import torch
import torch.nn as nn
import logging


class BaseAdapter(nn.Module):
    def __init__(self, cfg, model, optimizer):
        super().__init__()
        self.logger = logging.getLogger("TTA.adapter")
        self.cfg = cfg
        self.model = self.configure_model(model)

        params, param_names = self.collect_params(self.model)
        if len(param_names) == 0:
            self.optimizer = None
        else:
            self.optimizer = optimizer(params)

        self.steps = self.cfg.OPTIM.STEPS
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def forward_and_adapt(self, *args):
        raise NotImplementedError("implement forward_and_adapt by yourself!")

    def configure_model(self, model):
        raise NotImplementedError("implement configure_model by yourself!")

    def collect_params(self, model: nn.Module):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names

    def check_model(self, model):
        pass

    def before_tta(self, *args, **kwargs):
        pass

    @staticmethod
    def build_ema(model):
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model



@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
