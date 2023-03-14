import torch.optim as optim


def build_optimizer(cfg):
    def optimizer(params):
        if cfg.OPTIM.METHOD == 'Adam':
            return optim.Adam(params,
                              lr=cfg.OPTIM.LR,
                              betas=(cfg.OPTIM.BETA, 0.999),
                              weight_decay=cfg.OPTIM.WD)
        elif cfg.OPTIM.METHOD == 'SGD':
            return optim.SGD(params,
                             lr=cfg.OPTIM.LR,
                             momentum=cfg.OPTIM.MOMENTUM,
                             dampening=cfg.OPTIM.DAMPENING,
                             weight_decay=cfg.OPTIM.WD,
                             nesterov=cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    return optimizer
