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

"""
Script to measure training performance
"""
import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import optax
from absl import app, flags
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import transformer as transformer_lib
import utils
from train import TrainConfig, train_step


FLAGS = flags.FLAGS

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  api_version = FLAGS.api_version
  num_iters = FLAGS.num_iters

  train_fn = possible_api_versions[api_version]
  config = get_default_train_config()

  global_start = time.perf_counter()
  time_measures = train_fn(config, num_iters)
  global_elapsed = time.perf_counter() - global_start

  print(f"API Version: {api_version}")
  print(f"Time measurements (s):")
  for key, value in time_measures.items():
    print(f"- {key}: {value:.2f}")
  print(f"Total elapsed time (s): {global_elapsed:.2f}")


def get_default_train_config() -> TrainConfig:
  from configs.default import get_config

  return get_config()


def train_with_train_state(config: TrainConfig, num_iters: int) -> dict[str, float]:
  raise NotImplementedError()


def train_as_pytrees(*args, **kwargs) -> dict[str, float]:
  return generic_train_nnx(*args, **kwargs)


def train_with_hijax(*args, **kwargs) -> dict[str, float]:
  raise NotImplementedError()


def get_random_batch(batch_size, seq_len, vocab_size, npgen, data_sharding):
  seq = npgen.integers(0, vocab_size, size=(batch_size, seq_len + 1))
  pos = []
  segm = []
  for _ in range(batch_size):
    len_list = npgen.integers(5, seq_len // 3, size=(seq_len))
    temp_pos = []
    temp_segm, i = [], 0
    for le in len_list:
      temp_pos += list(range(le))
      temp_segm += [i] * le
      i += 1
      if len(temp_pos) >= seq_len:
        temp_pos = temp_pos[:seq_len]
        temp_segm = temp_segm[:seq_len]
        break
    pos.append(temp_pos)
    segm.append(temp_segm)
  pos = np.asarray(pos)
  segm = np.asarray(segm)
  batch = {
    "inputs": seq[:, :-1],
    "inputs_position": pos,
    "inputs_segmentation": segm,
    "targets": seq[:, 1:],
  }
  return jax.tree.map(
    lambda value: jax.make_array_from_process_local_data(data_sharding, value),
    batch,
  )

def generic_train_nnx(config: TrainConfig, num_iters: int):

  time_measures = {}

  devices_array = utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  data_sharding = NamedSharding(mesh, P(config.data_sharding))

  vocab_size = 30_000
  dtype = jnp.bfloat16 if config.use_bfloat16 else jnp.float32

  if config.transformer_name is not None:
    model_config = transformer_lib.TransformerConfig.from_version_name(
      config.transformer_name,
      num_embed=vocab_size,
      dtype=dtype,
    )
  else:
    assert config.transformer_params is not None
    model_config = transformer_lib.TransformerConfig.from_dict(
      **config.transformer_params,
      num_embed=vocab_size,
      dtype=dtype,
    )
  rngs = nnx.Rngs(params=config.seed, dropout=config.seed)
  npgen = np.random.default_rng(config.seed)

  print("- Create model and optimizer")
  start = time.perf_counter()
  with jax.set_mesh(mesh):
    model = transformer_lib.Transformer(model_config, rngs=rngs)
    optimizer = nnx.Optimizer(
      model,
      tx=optax.adamw(
        0.001,
        b1=0.9,
        b2=0.98,
        eps=1e-9,
        weight_decay=config.weight_decay,
      ),
      wrt=nnx.Param,
    )
  time_measures["model_optimizer_creation"] = time.perf_counter() - start

  jit_train_step = jax.jit(
    train_step,
    static_argnames=("label_smoothing", "pad_id"),
    donate_argnames=("model", "optimizer"),
  )

  batch_size = config.per_device_batch_size
  seq_len = config.max_target_length
  batch = get_random_batch(batch_size, seq_len, vocab_size, npgen, data_sharding)

  print("- Compile train_step")
  start = time.perf_counter()
  model, optimizer, rngs, _ = jit_train_step(
    model,
    optimizer,
    rngs,
    batch,
    0.0,  # label_smoothing
    0,  # pad_id
  )
  time_measures["compile train_step"] = time.perf_counter() - start

  print("- Start train_step measurements")
  batch = get_random_batch(batch_size, seq_len, vocab_size, npgen, data_sharding)
  start = time.perf_counter()
  for _ in range(num_iters):
    model, optimizer, rngs, metrics = jit_train_step(
      model,
      optimizer,
      rngs,
      batch,
      0.0,  # label_smoothing
      0,  # pad_id
    )
  metrics["loss"].block_until_ready()
  time_measures[f"train_step avg of {num_iters} trials"] = (time.perf_counter() - start) / num_iters

  return time_measures


if __name__ == '__main__':

  possible_api_versions = {
    "pytrees": train_as_pytrees,
    "train-state": train_with_train_state,
    "hijax": train_with_hijax,
  }
  list_api_versions = list(possible_api_versions.keys())

  flags.DEFINE_enum('api_version', None, list_api_versions, f'Flax API version')
  flags.mark_flags_as_required(['api_version'])

  flags.DEFINE_integer(
    'num_iters', 100,
    'Number of train_step iterations to measure',
    lower_bound=1,
  )

  jax.config.config_with_absl()
  app.run(main)
