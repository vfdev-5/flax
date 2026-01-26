# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline for image classification datasets."""

import os
import typing as tp

import albumentations as albu
import grain
import jax
import numpy as np
import tensorflow_datasets as tfds
from grain.python import MapTransform, MultiprocessingOptions
from grain.experimental import pick_performance_config

if tp.TYPE_CHECKING:
  from train import TrainConfig


Features = dict[str, tp.Any]

# image augmentations
img_size = 224

image_train_transforms = albu.Compose([
    albu.RandomResizedCrop(size=(img_size, img_size)),
    albu.HorizontalFlip(p=0.5),  # Horizontal random flip
    albu.RandomBrightnessContrast(p=0.4),  # Randomly changes the brightness and contrast
    albu.Normalize(),  # Normalize the image and cast to float
])

image_test_transforms = albu.Compose([
    albu.Resize(width=img_size, height=img_size),
    albu.Normalize(),  # Normalize the image and cast to float
])


def train_transforms(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return image_train_transforms(**x)


def test_transforms(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return image_test_transforms(**x)


def get_raw_dataset(dataset_name: str, split: str) -> grain.MapDataset:
  """Loads a dataset.

  Args:
    dataset_name: TFDS dataset name.
    split: Split to use. This must be the full split. We shard the split across
      multiple hosts and currently don't support sharding subsplits.
  """
  per_host_split = tfds.split_for_jax_process(split, drop_remainder=False)
  tfds_data_source = tfds.data_source(dataset_name, split=per_host_split)
  dataset = grain.MapDataset.source(tfds_data_source)
  return dataset


def move_to_devices(
    x: dict[str, np.ndarray], data_sharding: jax.sharding.Sharding
) -> dict[str, jax.Array]:
  return jax.tree.map(
    lambda value: jax.make_array_from_process_local_data(data_sharding, value), x
  )

# -----------------------------------------------------------------------------
# Main dataset prep routines.
# -----------------------------------------------------------------------------
def preprocess_data(
    dataset: grain.MapDataset,
    transforms: tp.Callable,
    shuffle: bool,
    num_epochs: int | None = 1,
    batch_size: int = 256,
    drop_remainder: bool = True,
    seed: int = 41,
    prefetch_num_workers: int | None = None,
    data_sharding: jax.sharding.Sharding | None = None,
) -> grain.IterDataset:
  """Shuffle and batch/pack the given dataset."""

  if shuffle:
    dataset = dataset.shuffle(seed=seed)

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.to_iter_dataset()
  dataset = dataset.map(transforms)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

  if prefetch_num_workers is None:
    performance_config = pick_performance_config(
        ds=dataset,
        ram_budget_mb=1024,
        max_workers=os.cpu_count() // 2,
        max_buffer_size=None
    )
    mp_optons = performance_config.multiprocessing_options
  else:
    mp_optons = MultiprocessingOptions(num_workers=prefetch_num_workers)
  dataset = dataset.mp_prefetch(mp_optons)

  # Move data to jax array
  if data_sharding is not None:
    dataset = dataset.map(lambda x: move_to_devices(x, data_sharding=data_sharding))
  else:
    dataset = dataset.map(lambda x: jax.tree.map(jax.numpy.asarray, x))

  return dataset


def get_datasets(
    config: "TrainConfig",
    *,
    data_sharding: jax.sharding.Sharding | None = None,
):
  """Load and return dataset of batched examples for use during training."""
  train_data = get_raw_dataset(config.dataset_name, split="train")
  eval_data = get_raw_dataset(config.eval_dataset_name, split=config.eval_split)

  if data_sharding is not None:
    # We set n_devices to the number of local devices
    # as we use then jax.make_array_from_process_local_data
    # to create a large batch = per_device_batch_size * num_local_devices * num_procs
    n_devices = len(data_sharding.addressable_devices)
  else:
    n_devices = 1

  batch_size = config.per_device_batch_size * n_devices
  if config.eval_per_device_batch_size > 0:
    eval_batch_size = config.eval_per_device_batch_size * n_devices
  else:
    eval_batch_size = batch_size

  train_ds = preprocess_data(
      train_data,
      transforms=train_transforms,
      shuffle=True,
      num_epochs=None,
      batch_size=batch_size,
      seed=config.seed,
      prefetch_num_workers=config.prefetch_num_workers,
      data_sharding=data_sharding,
  )

  eval_ds = preprocess_data(
      eval_data,
      transforms=test_transforms,
      shuffle=False,
      batch_size=eval_batch_size,
      prefetch_num_workers=config.prefetch_num_workers,
      data_sharding=data_sharding,
  )

  return train_ds, eval_ds
