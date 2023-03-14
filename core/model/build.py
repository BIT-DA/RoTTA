from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


def build_model(cfg):
    if cfg.CORRUPTION.DATASET in ["cifar10", "cifar100"]:
        base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                                cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    else:
        raise NotImplementedError()

    return base_model
