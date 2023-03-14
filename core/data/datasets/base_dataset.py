from ...utils.utils import check_isfile
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import List
import torchvision.transforms as T
from PIL import Image


INTERPOLATION_MODES = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
}


class DatumBase:
    def __init__(self, img=None,  label=0, domain=0, classname=""):
        self._img = img
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def img(self):
        return self._img

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatumList(DatumBase):
    def __init__(self, img="", label=0, domain=0, classname=""):
        assert isinstance(img, str)
        assert check_isfile(img)
        super().__init__(img, label, domain, classname)


class DatumRaw(DatumBase):
    def __init__(self, img=None, label=0, domain=0, classname=""):
        assert isinstance(img, torch.Tensor), f"error type for DatumRaw: {type(img)}"
        super().__init__(img, label, domain, classname)


class TTADatasetBase(TorchDataset):
    dataset_dir = ""
    domains = []

    def __init__(self, cfg, data_source: List[DatumBase]):
        self.cfg = cfg
        self.data_source = data_source
        self.datum_type = type(data_source[0])

        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, item):
        if self.datum_type == DatumList:
            return self.__get_from_path(item)
        elif self.datum_type == DatumRaw:
            return self.__get_from_raw(item)
        else:
            raise RuntimeError(f"error type of datum: {self.datum_type}")

    def __get_from_path(self, index):
        item = self.data_source[index]

        ret_data = {
            "label": item.label,
            "domain": item.domain
        }

        img = Image.open(item.img).convert("RGB")

        ret_data["image"] = self.to_tensor(img)

        return ret_data

    def __get_from_raw(self, index):
        item = self.data_source[index]

        ret_data = {
            "label": item.label,
            "domain": item.domain,
            "image": item.img
        }

        return ret_data

