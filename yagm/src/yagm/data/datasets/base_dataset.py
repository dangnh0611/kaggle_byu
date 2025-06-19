import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):

    def __init__(self, cfg, stage="train", cache=None):
        self.cfg = cfg
        self.CACHE = cache
        logger.info("INITIALIZING %s dataset", stage)
        assert stage in ["train", "val", "test"]
        self.stage = stage

    def __len__(self):
        raise NotImplementedError

    @property
    def getitem_as_batch(self):
        """
        Whether __getitem__() return a batch instead of single samples.
        In that case, __getitem__(self, idxs) accept a list of indices as input arg
        """
        return False

    def compute_sampling_weights(self):
        return [1.0] * len(self)

    def __getitem__(self, idx):
        """
        Generate one batch of data.
        """
        raise NotImplementedError

    @classmethod
    def load_cache(cls, cfg):
        return {}
