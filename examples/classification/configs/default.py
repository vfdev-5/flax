# Copyright 2026 The Flax Authors.
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

"""Default Hyperparameter configuration."""

import dataclasses

from train import TrainConfig


@dataclasses.dataclass(unsafe_hash=True)
class Config:
  # Name of TFDS image classification dataset to use.
  # Imagenette: https://www.tensorflow.org/datasets/catalog/imagenette
  dataset_name: str = 'imagenette'
  # Optional name of TFDS image classification dataset to use for evaluation.
  eval_dataset_name: str = 'imagenette'
  # Optional name of TFDS split to use for evaluation.
  eval_split: str = 'validation'
  # Per device batch size for training.
  per_device_batch_size: int = 32
  # Per device batch size for training.
  eval_per_device_batch_size: int = 32
  # Grain prefetch number of workers.
  prefetch_num_workers: int | None = 4

  # Number of steps to take during training.
  num_train_steps: int = 5_000
  # Number of steps to take during evaluation.
  num_eval_steps: int = 1000

  # TODO: Number of steps to generate predictions.
  # -1 will use the whole eval dataset.
  num_predict_steps: int = 50

  # Base learning rate.
  learning_rate: float = 0.0016
  # Linear learning rate warmup.
  warmup_steps: int = 400
  # Decay factor for AdamW style weight decay.
  weight_decay: float = 0.1

  # Image classification model name from bonsai: https://github.com/jax-ml/bonsai/blob/main/README.md
  model_name: str = "ConvNeXt"
  model_config_name: str = "convnext_base_224"

  # Whether to save model checkpoints.
  save_checkpoints: bool = True
  # Whether to restore from existing model checkpoints.
  restore_checkpoints: bool = True
  # Save a checkpoint every these number of steps.
  checkpoint_every_steps: int = 1000
  # Frequency of eval during training, e.g. every 2_000 steps.
  eval_every_steps: int = 500
  # Use bfloat16 mixed precision training instead of float32.
  use_bfloat16: bool = False
  # Integer for PRNG random seed.
  seed: int = 0

  # Parallelism
  mesh_axes: tuple[str, ...] = ('data', 'fsdp', 'tensor')
  data_sharding: tuple[str, ...] = ('data', 'fsdp')

  data_parallelism: int = -1
  fsdp_parallelism: int = 1
  tensor_parallelism: int = 1

  with_profiler_step_trace: bool = False


def get_config() -> TrainConfig:
  """Get the default hyperparameter configuration."""
  config = Config()
  return TrainConfig(**dataclasses.asdict(config))
