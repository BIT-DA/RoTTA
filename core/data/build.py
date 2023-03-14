from .datasets.common_corruption import CorruptionCIFAR
from .ttasampler import build_sampler
from torch.utils.data import DataLoader
from ..utils.result_precess import AvgResultProcessor


def build_loader(cfg, ds_name, all_corruptions, all_severity):
    if ds_name == "cifar10" or ds_name == "cifar100":
        dataset_class = CorruptionCIFAR
    else:
        raise NotImplementedError(f"Not Implement for dataset: {cfg.CORRUPTION.DATASET}")

    ds = dataset_class(cfg, all_corruptions, all_severity)
    sampler = build_sampler(cfg, ds.data_source)

    loader = DataLoader(ds, cfg.TEST.BATCH_SIZE, sampler=sampler, num_workers=cfg.LOADER.NUM_WORKS)

    result_processor = AvgResultProcessor(ds.domain_id_to_name)

    return loader, result_processor
