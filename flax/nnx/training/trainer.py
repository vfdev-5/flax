import dataclasses
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

import jax
import flax.nnx as nnx
import orbax.checkpoint as ocp
from absl import logging
from clu import platform
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint.path import atomicity


@dataclasses.dataclass(slots=True)
class TrainConfig:
  # Integer for PRNG random seed.
  seed: int

  # Number of steps to take during training.
  num_train_steps: int
  # Number of steps to take during evaluation.
  num_eval_steps: int

  # Whether to save model checkpoints.
  save_checkpoints: bool = True
  # Whether to restore from existing model checkpoints.
  restore_checkpoints: bool = True
  # Save a checkpoint every these number of steps.
  checkpoint_every_steps: int = 200
  # Frequency of eval during training, e.g. every 1_000 steps.
  eval_every_steps: int = 100

  # Parallelism
  mesh_config: dict[str, int] | None = None

  # TODO: recompute -1 in mesh_config at post init level



class Trainer(ABC):
  def __init__(self, *, config: TrainConfig):
    try:
      jax.distributed.initialize()
    except ValueError:
      # On single GPU host above command can raise ValueError: coordinator_address should be defined.
      # This is fine.
      pass

    self.config = config

    self.model = None
    self.optimizer = None
    self.mesh = None
    self.train_iter = None
    self.eval_iter = None
    self.rngs = None

  @abstractmethod
  def initialize(self) -> tuple[nnx.Module, nnx.Optimizer, nnx.Rngs]:
    pass

  @abstractmethod
  def train_step(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      rngs: nnx.Rngs,
      batch: Any,
      train_metrics: nnx.Metric | None = None,
  ):
    pass

  @abstractmethod
  def eval_step(
      self,
      model: nnx.Module,
      batch: Any,
      eval_metrics: nnx.Metric,
  ):
    pass

  def make_mesh(self) -> jax.Mesh | None:
    if self.config.mesh_config is not None:
      return jax.make_mesh(
        tuple(self.config.mesh_config.keys()),
        tuple(self.config.mesh_config.values()),
      )
    return None

  def _train(self):
    pass

  def _evaluate(self):
    pass

  @abstractmethod
  def get_train_iter(self):
    pass

  @abstractmethod
  def get_eval_iter(self):
    pass

  def run(self) -> None:
    logging.info(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    logging.info(f'JAX devices: {jax.devices()}')

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f'process_index: {jax.process_index()}, '
        f'process_count: {jax.process_count()}'
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, self.workdir, 'workdir'
    )

    initial_mesh = jax.sharding.get_mesh()
    self.mesh = self.make_mesh()
    if self.mesh is not None:
      jax.set_mesh(self.mesh)

    self.model, self.optimizer, self.rngs = self.initialize()

    checkpoint_mngr = ocp.CheckpointManager(
      checkpoint_path,
      options=ocp.CheckpointManagerOptions(
        preservation_policy=preservation_policy_lib.LatestN(1),
        temporary_path_class=atomicity.CommitFileTemporaryPath
      )
    )



    if self.mesh is not None:
      jax.set_mesh(initial_mesh)
