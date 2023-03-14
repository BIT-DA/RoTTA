from .base_adapter import BaseAdapter
from .rotta import RoTTA


def build_adapter(cfg) -> type(BaseAdapter):
    if cfg.ADAPTER.NAME == "rotta":
        return RoTTA
    else:
        raise NotImplementedError("Implement your own adapter")

