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

# Copied over from MaxText
# (https://github.com/google/maxtext/blob/main/MaxText/max_utils.py).
"""Provides utilities for training the Flax gemma example."""

import logging
from typing import Any

import jax
from jax.experimental import mesh_utils
import numpy as np


# Mesh utils.
# -----------------------------------------------------------------------------


def create_device_mesh(config: Any):
  """Creates a device mesh with each slice in its own data parallel group.

  If there is only one slice, uses two replicas.

  Args:
    config: The training configuration.
  Returns:
    The device mesh.
  """
  devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max([d.slice_index for d in devices])
  except AttributeError:
    num_slices = 1
  num_devices_per_slice = num_devices // num_slices
  logging.info(f'Devices: {devices}')  # pylint: disable=logging-fstring-interpolation
  logging.info(f'Number of devices: {num_devices}')  # pylint: disable=logging-fstring-interpolation

  multi_slice_env = hasattr(jax.devices()[0], 'slice_index')

  dcn_parallelism = [
      config.dcn_data_parallelism,
      config.dcn_fsdp_parallelism,
      config.dcn_tensor_parallelism,
  ]
  ici_parallelism = [
      config.ici_data_parallelism,
      config.ici_fsdp_parallelism,
      min(config.ici_tensor_parallelism, max(num_devices_per_slice // 2, 1)),
  ]

  # Find possible unspecified parallelisms
  dcn_parallelism = fill_unspecified_mesh_axes(
      dcn_parallelism, num_slices, 'DCN'
  )
  ici_parallelism = fill_unspecified_mesh_axes(
      ici_parallelism, num_devices_per_slice, 'ICI'
  )

  if multi_slice_env:
    mesh = mesh_utils.create_hybrid_device_mesh(
        ici_parallelism, dcn_parallelism
    )
  else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism)

  logging.info(f'Decided on mesh: {mesh}')  # pylint: disable=logging-fstring-interpolation
  logging.info(f'Mesh shape: {mesh.shape}')  # pylint: disable=logging-fstring-interpolation

  return mesh


def fill_unspecified_mesh_axes(
    parallelism_vals, target_product, parallelism_type
):
  """Evaluates unspecified DCN/ICI parallelism values."""
  if -1 in parallelism_vals:
    assert parallelism_vals.count(-1) == 1, (
        f'Found unspecified values (-1) for more than one {parallelism_type}   '
        '   parallelism axis. At most one axis can be unspecified.'
    )

    determined_val = target_product / np.prod(parallelism_vals) * -1

    assert determined_val >= 1 and determined_val.is_integer, (
        'Unspecified value unable to be determined with the given     '
        f' {parallelism_type} parallelism values'
    )

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = 'slices' if parallelism_type == 'DCN' else 'devices per slice'

  assert np.prod(parallelism_vals) == target_product, (
      f'Number of {target_type} {target_product} does not match    the product'
      f' of the {parallelism_type} parallelism {np.prod(parallelism_vals)}'
  )

  return parallelism_vals
