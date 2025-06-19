import gc
import logging
from typing import Any, Dict

import hydra
import torch
from lightning import LightningDataModule
from lightning.fabric.utilities.seed import pl_worker_init_function
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaseDataModule(LightningDataModule):

    def __init__(self, cfg, cache={}):
        super().__init__()
        self.cfg = cfg

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.cache = cache

    def load_cache(self):
        dataset_cls = hydra.utils.get_class(self.cfg.data._dataset_target_)
        if hasattr(dataset_cls, "load_cache"):
            return dataset_cls.load_cache(self.cfg)
        return {}

    def set_cache(self, cache):
        self.cache = cache

    def clear_cache(self):
        self.cache = {}
        gc.collect()

    def prepair_data(self):
        pass

    def setup_dataset(self, stage="train"):
        assert stage in ["train", "val", "test", "predict"]
        self.dataset_cls = hydra.utils.get_class(self.cfg.data._dataset_target_)
        return self.dataset_cls(self.cfg, stage=stage, cache=self.cache)

    def setup(self, stage: str):
        if stage == "fit":
            if self.train_dataset is None:
                self.train_dataset = self.setup_dataset(stage="train")
            if self.val_dataset is None:
                do_validation = (self.cfg.trainer.limit_val_batches is None) or (
                    self.cfg.trainer.limit_val_batches > 0
                )
                if do_validation:
                    self.val_dataset = self.setup_dataset(stage="val")
        elif stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = self.setup_dataset(stage="val")
        elif stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self.setup_dataset(stage="test")
        elif stage == "predict":
            if self.predict_dataset is None:
                self.predict_dataset = self.setup_dataset(stage="predict")
        else:
            raise AssertionError

    @property
    def worker_init_fn(self):

        def _worker_init_fn(worker_id: int, rank=None):
            if self.cfg.loader.pytorch_num_threads is not None:
                # when using Dataloader with num_workers > 0,
                # default behavior of Pytorch reduce the number of threads to 1 (usually from 32),
                # possibly to prevent overloading the CPU by too much threads.
                # but, sometime we still need more Pytorch threads to speedup CPU operations,
                # such as torch.nn.functional.grid_resample(), which could take seconds to finish with 1 thread.
                assert self.cfg.loader.pytorch_num_threads > 0
                cur_pytorch_num_threads = torch.get_num_threads()
                torch.set_num_threads(self.cfg.loader.pytorch_num_threads)
                assert self.cfg.loader.pytorch_num_threads == torch.get_num_threads()
                logger.debug(
                    "Change Pytorch number of threads from %d to %d in dataloader worker %d",
                    cur_pytorch_num_threads,
                    torch.get_num_threads(),
                    worker_id,
                )

            pl_worker_init_function(worker_id, rank)
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            if hasattr(dataset, "worker_init_fn"):
                logger.debug("Trigger dataset's worker_init_fn()")
                dataset.worker_init_fn(worker_id, rank)

        return _worker_init_fn

    def train_dataloader(self) -> Any:
        dataset = self.train_dataset
        if hasattr(dataset, "getitem_as_batch"):
            getitem_as_batch = dataset.getitem_as_batch
        else:
            getitem_as_batch = False

        # build sampler
        if getattr(self.cfg.loader, "sampler", None) is not None:
            sampler_cls = hydra.utils.get_class(self.cfg.loader.sampler._target_)
            sampler_kwargs = dict(self.cfg.loader.sampler)
            sampler_kwargs.pop("_target_")
            sampler = sampler_cls(dataset, **sampler_kwargs)
        else:
            sampler = None

        if getitem_as_batch:
            from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

            if sampler is None:
                # default behavior of torch.utils.data.DataLoader
                sampler = RandomSampler(
                    dataset, replacement=False, num_samples=None, generator=None
                )
            # dataset's __getitem__(self, idxs) where idxs is a list of indices
            # this may speedup when e.g tokenizer a batch of strings
            # instead of tokenize each idividual string then collate to a batch
            sampler = BatchSampler(
                sampler, self.cfg.loader.train_batch_size, self.cfg.loader.drop_last
            )

        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = None

        return DataLoader(
            dataset,
            batch_size=None if getitem_as_batch else self.cfg.loader.train_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            batch_sampler=None,
            num_workers=self.cfg.loader.train_num_workers,
            collate_fn=collate_fn,
            pin_memory=self.cfg.loader.pin_memory,
            drop_last=False if getitem_as_batch else self.cfg.loader.drop_last,
            worker_init_fn=self.worker_init_fn,
            prefetch_factor=None,
            persistent_workers=self.cfg.loader.persistent_workers,
            pin_memory_device="",
        )

    # def train_dataloader(self) -> torch.Any:
    #     from src.utils.sampler import WeightedInfiniteSampler

    #     if hasattr(self.train_dataset, "compute_sampling_weights"):
    #         weights = self.train_dataset.compute_sampling_weights()
    #     else:
    #         # uniform distribution
    #         weights = [1.0] * len(self.train_dataset)

    #     samples_per_epoch = getattr(self.cfg.loader, "samples_per_epoch", None)
    #     if samples_per_epoch is not None and samples_per_epoch > 0:
    #         pass
    #     elif self.cfg.loader.steps_per_epoch > 0:
    #         samples_per_epoch = (
    #             self.cfg.loader.steps_per_epoch
    #             * self.cfg.loader.train_batch_size
    #             * self.cfg.trainer.accumulate_grad_batches
    #         )
    #     else:
    #         samples_per_epoch = len(self.train_dataset)

    #     sampler = WeightedInfiniteSampler(
    #         size=samples_per_epoch,
    #         weights=weights,
    #         seed=self.cfg.seed,
    #         rank=0,
    #         world_size=1,
    #     )

    #     return DataLoader(
    #         self.train_dataset,
    #         sampler=sampler,
    #         batch_size=self.cfg.loader.train_batch_size,
    #         num_workers=self.cfg.loader.train_num_workers,
    #         pin_memory=self.cfg.loader.pin_memory,
    #         drop_last=self.cfg.loader.drop_last,
    #         persistent_workers=self.cfg.loader.persistent_workers,
    #         pin_memory_device="",
    #     )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = self.val_dataset
        if hasattr(dataset, "getitem_as_batch"):
            getitem_as_batch = dataset.getitem_as_batch
        else:
            getitem_as_batch = False
        sampler = None
        if getitem_as_batch:
            from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

            if sampler is None:
                # default behavior of torch.utils.data.DataLoader
                sampler = SequentialSampler(dataset)
            # dataset's __getitem__(self, idxs) where idxs is a list of indices
            # this may speedup when e.g tokenizer a batch of strings
            # instead of tokenize each idividual string then collate to a batch
            sampler = BatchSampler(sampler, self.cfg.loader.val_batch_size, False)

        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = None

        print("\n\n\nLEN VAL DATASET:", len(dataset))

        return DataLoader(
            dataset,
            batch_size=None if getitem_as_batch else self.cfg.loader.val_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.loader.val_num_workers,
            collate_fn=collate_fn,
            pin_memory=self.cfg.loader.pin_memory,
            drop_last=False,
            worker_init_fn=self.worker_init_fn,
            prefetch_factor=None,
            persistent_workers=self.cfg.loader.persistent_workers,
            pin_memory_device="",
        )

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = self.test_dataset
        if hasattr(dataset, "getitem_as_batch"):
            getitem_as_batch = dataset.getitem_as_batch
        else:
            getitem_as_batch = False
        sampler = None
        if getitem_as_batch:
            from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

            if sampler is None:
                # default behavior of torch.utils.data.DataLoader
                sampler = SequentialSampler(dataset)
            # dataset's __getitem__(self, idxs) where idxs is a list of indices
            # this may speedup when e.g tokenizer a batch of strings
            # instead of tokenize each idividual string then collate to a batch
            sampler = BatchSampler(sampler, self.cfg.loader.val_batch_size, False)

        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = None

        if hasattr(dataset, "worker_init_fn"):
            worker_init_fn = dataset.worker_init_fn
        else:
            worker_init_fn = None

        return DataLoader(
            dataset,
            batch_size=None if getitem_as_batch else self.cfg.loader.val_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.loader.val_num_workers,
            collate_fn=collate_fn,
            pin_memory=self.cfg.loader.pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            prefetch_factor=None,
            persistent_workers=self.cfg.loader.persistent_workers,
            pin_memory_device="",
        )

    def predict_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = self.predict_dataset
        if hasattr(dataset, "getitem_as_batch"):
            getitem_as_batch = dataset.getitem_as_batch
        else:
            getitem_as_batch = False
        sampler = None
        if getitem_as_batch:
            from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

            if sampler is None:
                # default behavior of torch.utils.data.DataLoader
                sampler = SequentialSampler(dataset)
            # dataset's __getitem__(self, idxs) where idxs is a list of indices
            # this may speedup when e.g tokenizer a batch of strings
            # instead of tokenize each idividual string then collate to a batch
            sampler = BatchSampler(sampler, self.cfg.loader.val_batch_size, False)

        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = None

        if hasattr(dataset, "worker_init_fn"):
            worker_init_fn = dataset.worker_init_fn
        else:
            worker_init_fn = None

        return DataLoader(
            dataset,
            batch_size=None if getitem_as_batch else self.cfg.loader.val_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.loader.val_num_workers,
            collate_fn=collate_fn,
            pin_memory=self.cfg.loader.pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            prefetch_factor=None,
            persistent_workers=self.cfg.loader.persistent_workers,
            pin_memory_device="",
        )

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule state_dict.

        Args:
            state_dict: the datamodule state returned by ``state_dict``.
        """
        pass

    def teardown(self, stage: str) -> None:
        self.clear_cache()
        return super().teardown(stage)
