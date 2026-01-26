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

"""Language Modeling example.

This script trains a Transformer on a LM1B dataset.
"""

# pytype: disable=wrong-arg-count
# pytype: disable=attribute-error
import contextlib
import dataclasses
from pathlib import Path

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import nnx
import input_pipeline
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint.path import atomicity


@dataclasses.dataclass(unsafe_hash=True)
class TrainConfig:
  """Configuration for training a gemma model."""

  # Name of TFDS image classification dataset to use.
  # List of datasets: https://www.tensorflow.org/datasets/catalog/overview
  dataset_name: str
  # Optional name of TFDS image classification dataset to use for evaluation.
  eval_dataset_name: str
  # Optional name of TFDS split to use for evaluation.
  eval_split: str
  # Per device batch size for training.
  per_device_batch_size: int
  # Per device batch size for training.
  eval_per_device_batch_size: int
  # Grain prefetch number of workers.
  prefetch_num_workers: int | None

  # Number of steps to take during training.
  num_train_steps: int
  # Number of steps to take during evaluation.
  # Large enough to evaluate all samples: 306_688 / (32 * 8) = 1198
  num_eval_steps: int

  # TODO: Number of steps to generate predictions.
  # -1 will use the whole eval dataset.
  num_predict_steps: int

  # Base learning rate.
  learning_rate: float
  # Linear learning rate warmup.
  warmup_steps: int
  # Decay factor for AdamW style weight decay.
  weight_decay: float

  # Image classification model name from bonsai: https://github.com/jax-ml/bonsai/blob/main/README.md
  model_name: str
  model_config_name: str

  # Whether to save model checkpoints.
  save_checkpoints: bool
  # Whether to restore from existing model checkpoints.
  restore_checkpoints: bool
  # Save a checkpoint every these number of steps.
  checkpoint_every_steps: int
  # Frequency of eval during training, e.g. every 1_000 steps.
  eval_every_steps: int
  # Use bfloat16 mixed precision training instead of float32.
  use_bfloat16: bool
  # Integer for PRNG random seed.
  seed: int

  # Parallelism
  mesh_axes: tuple[str, ...]
  data_sharding: tuple[str | tuple[str], ...]

  data_parallelism: int = 1
  fsdp_parallelism: int = -1
  tensor_parallelism: int = 1

  # Profiling
  with_profiler_step_trace: bool = False

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)

  def __post_init__(self):
    axis_shapes = [self.data_parallelism, self.fsdp_parallelism, self.tensor_parallelism]
    assert axis_shapes.count(-1) in (0, 1), (
      f'Found unspecified values (-1) for more than one parallelism axis. '
      'At most one axis can be unspecified.'
    )

  def get_mesh_shape(self, num_devices: int) -> tuple[int, int, int]:
    axis_shapes = [self.data_parallelism, self.fsdp_parallelism, self.tensor_parallelism]
    count = np.prod(axis_shapes)
    if count < 0:
      axis_shapes[axis_shapes.index(-1)] = int(num_devices / (-count))
    else:
      assert count == num_devices
    return tuple(axis_shapes)


def rsqrt_schedule(
    init_value: float,
    shift: int = 0,
):
  """Applies a reverse square-root schedule.

  The reverse square root schedule is simply `lr = init_value / sqrt(step)`.

  Args:
    init_value: Base learning rate (before applying the rsqrt schedule).
    shift: How many steps the rsqrt should be shifted. Shifting the rsqrt
      schedule makes it less steep in the beginning (close to 0).

  Returns:
    A schedule that applies the reverse square root.
  """

  def schedule(count):
    return init_value * (count + shift) ** -0.5 * shift**0.5

  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules(
      [
          optax.linear_schedule(
              init_value=0,
              end_value=learning_rate,
              transition_steps=warmup_steps,
          ),
          rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
      ],
      boundaries=[warmup_steps],
  )


@jax.jit
def compute_metrics_summary(
  metrics_list: list[dict[str, jax.Array]],
) -> dict[str, jax.Array]:

  metrics_dict = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *metrics_list)
  metrics_sums = jax.tree.map(jnp.sum, metrics_dict)
  denominator = metrics_sums.pop('denominator')
  summary = jax.tree.map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
  return summary


def get_bonsai_model(config: TrainConfig, *, rngs: nnx.Rngs) -> nnx.Module:
  from bonsai import models

  if not hasattr(models, config.model_name):
    raise ValueError(
      f"Model name: '{config.model_name}' is not found in Bonsai models. "
      "Possible values: ConvNeXt, ResNet, VGG, DenseNet, ..."
    )

  model_cls = getattr(models, config.model_name)
  model_config_cls = getattr(models, f"{config.model_name}Config")
  config = getattr(model_config_cls, config.model_config_name)()
  return model_cls(config, rngs=rngs)


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    batch: dict[str, jax.Array],
    train_metrics: nnx.Metric,
) -> tuple[nnx.Module, nnx.Optimizer, nnx.Rngs, nnx.Metric]:
  images, labels = batch["image"], batch["label"]
  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

  def loss_fn(params, rngs):
    """loss function used for training."""
    module = nnx.merge(graphdef, params, nondiff)
    logits = module(images, rngs=rngs)
    loss_per_sample = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    mean_loss = loss_per_sample.mean()
    return mean_loss, (loss_per_sample, logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, (loss_per_sample, logits)), grads = grad_fn(params, rngs.fork())
  optimizer.update(model, grads)

  train_metrics.update(loss=loss_per_sample, logits=logits, labels=labels)
  return model, optimizer, rngs, train_metrics


def eval_step(
    model: nnx.Module,
    batch: dict[str, jax.Array],
    eval_metrics: nnx.Metric,
) -> nnx.Metric:
  """Calculate evaluation metrics on a batch."""
  images, labels = batch["image"], batch["label"]
  inspect_sharding(images, tag="image")
  inspect_sharding(labels, tag="label")
  logits = model(images)

  loss_per_sample = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
  eval_metrics.update(loss=loss_per_sample, logits=logits, labels=labels)
  return eval_metrics


def evaluate(
    *,
    jit_eval_step,
    model: nnx.Module,
    eval_ds: grain.IterDataset,
    num_eval_steps: int,
    eval_metrics: nnx.Metric | None = None,
) -> dict[str, jax.Array]:
  """Evaluate the target an return a dictionary with the metrics."""
  logging.info('Gathering evaluation metrics.')
  if eval_metrics is None:
      eval_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        accuracy=nnx.metrics.Accuracy(),
      )

  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  for _, eval_batch in zip(range(num_eval_steps), eval_iter):
    eval_metrics = jit_eval_step(model, eval_batch, eval_metrics)

  return eval_metrics.compute()


def inspect_sharding(x, tag=""):
  print(tag, x.sharding if hasattr(x, "sharding") else jax.typeof(x))
  info = x.sharding.devices_indices_map(tuple(x.shape))
  for key, value in info.items():
    print(f" - Device {key.id}: {value}")


def train_and_evaluate(config: TrainConfig, workdir: str, chpt_bucket: str | None = None):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  workdir = Path(workdir).absolute().resolve()
  workdir.mkdir(parents=True, exist_ok=True)

  checkpoint_path = workdir / "checkpoints" if chpt_bucket is None else chpt_bucket

  workdir = str(workdir)

  # Mesh definition
  mesh = jax.make_mesh(config.get_mesh_shape(len(jax.devices())), config.mesh_axes)

  # Load Dataset
  # ---------------------------------------------------------------------------
  logging.info('Initializing dataset.')
  data_sharding = NamedSharding(mesh, jax.P(config.data_sharding))

  train_ds, eval_ds = input_pipeline.get_datasets(config=config, data_sharding=data_sharding)
  # train_ds and eval_ds provide batches of type dict:
  # {"image": jax.Array[B, S, S, 3], "label": jax.Array[B]}
  # where B = config.per_device_batch_size, S = input_pipeline.img_size

  train_iter = iter(train_ds)

  rngs = nnx.Rngs(params=config.seed, dropout=config.seed)
  logging.info('Initializing model, optimizer, and step functions.')
  # Build Model and Optimizer
  start_step = 0
  learning_rate_fn = create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  with jax.set_mesh(mesh):
    model = get_bonsai_model(config, rngs=rngs)
    optimizer = nnx.Optimizer(
        model,
        tx=optax.adamw(
            learning_rate_fn,
            b1=0.9,
            b2=0.98,
            eps=1e-9,
            weight_decay=config.weight_decay,
        ),
        wrt=nnx.Param,
    )
    # get model view for evaluation
    try:
      eval_model = nnx.view(model, deterministic=True, use_running_average=True)
    except ValueError:
      # Case if model does not have Dropout, BatchNorm etc
      eval_model = model

  # Report number of parameters in the model:
  flat_state = nnx.to_flat_state(nnx.state(model))
  num_params = {str(key): param.size for key, param in flat_state}
  total_num_params = sum([value for _, value in num_params.items()])
  logging.info(
    "\nModel Number of Parameters:\n"
    f"- Total (M): {total_num_params / 1_000_000}\n"
  )

  checkpoint_mngr = ocp.CheckpointManager(
    checkpoint_path,
    options=ocp.CheckpointManagerOptions(
      preservation_policy=preservation_policy_lib.LatestN(1),
      temporary_path_class=atomicity.CommitFileTemporaryPath
    )
  )

  if config.restore_checkpoints and checkpoint_mngr.latest_step() is not None:
    # Restore unreplicated optimizer + model state from last checkpoint.
    target = {
      "model": nnx.state(model),
      "optimizer": nnx.state(optimizer),
      "step": 0,
    }
    checkpoint = checkpoint_mngr.restore(
      checkpoint_mngr.latest_step(),
      args=ocp.args.StandardRestore(target),
    )
    nnx.update(model, checkpoint["model"])
    nnx.update(optimizer, checkpoint["optimizer"])
    start_step = checkpoint["step"] + 1  # Add +1 to skip saving again the same step

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  if start_step == 0:
    writer.write_hparams(dataclasses.asdict(config))

  train_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    accuracy=nnx.metrics.Accuracy(),
  )
  eval_metrics = nnx.MultiMetric(
      loss=nnx.metrics.Average('loss'),
      accuracy=nnx.metrics.Accuracy(),
  )

  jit_train_step = jax.jit(
      train_step,
      donate_argnames=("model", "optimizer"),
  )

  # jit_eval_step = jax.jit(eval_step)
  jit_eval_step = eval_step

  # Main Train Loop
  # ---------------------------------------------------------------------------
  logging.info('Starting training loop.')
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(logdir=workdir, num_profile_steps=10),
    ]
  with metric_writers.ensure_flushes(writer), jax.set_mesh(mesh):
    for step in range(start_step, config.num_train_steps):
      is_last_step = step == config.num_train_steps - 1

      maybe_profiler_step_trace = (
        jax.profiler.StepTraceAnnotation('train', step_num=step)
        if config.with_profiler_step_trace else
        contextlib.suppress()
      )

      with maybe_profiler_step_trace:
        with report_progress.timed('data'):
          batch = next(train_iter)

        with report_progress.timed('train_step'):
          model, optimizer, rngs, train_metrics = jit_train_step(
              model,
              optimizer,
              rngs,
              batch,
              train_metrics
          )

      # Quick indication that training is happening.
      if step < 20:
        logging.info(
          "Finished training step %d. Batch size: %d, Loss: %.5f, LR: %.5f",
          step,
          len(batch['image']),
          train_metrics.compute()["loss"],
          learning_rate_fn(step + 1),
        )
      for h in hooks:
        h(step)

      # Periodic metric handling.
      # if (step > 0 and step % config.eval_every_steps == 0) or is_last_step:
      if (step % config.eval_every_steps == 0) or is_last_step:
        with report_progress.timed('training_metrics'):
          logging.info('Gathering training metrics.')
          summary = train_metrics.compute()
          summary = {'train_' + k: v for k, v in summary.items()}
          writer.write_scalars(step, summary)

        with report_progress.timed('eval'):
          eval_results = evaluate(
              jit_eval_step=jit_eval_step,
              model=eval_model,
              eval_ds=eval_ds,
              num_eval_steps=config.num_eval_steps,
              eval_metrics=eval_metrics,
              rngs=rngs,
          )
          writer.write_scalars(
              step, {'eval_' + k: v for k, v in eval_results.items()}
          )

      # Save a checkpoint on one host after every checkpoint_freq steps.
      save_checkpoint = (
          (step > 0 and step % config.checkpoint_every_steps == 0) or is_last_step
      )
      if config.save_checkpoints and save_checkpoint:
        logging.info('Saving checkpoint step %d.', step)
        with report_progress.timed('checkpoint'):
          checkpoint = {
            "model": nnx.state(model),
            "optimizer": nnx.state(optimizer),
            "step": step,
          }
          checkpoint_mngr.save(step, args=ocp.args.StandardSave(checkpoint))

  checkpoint_mngr.wait_until_finished()