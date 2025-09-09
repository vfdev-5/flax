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

import jax
import numpy as np
from absl.testing import absltest
from jax import numpy as jnp
from jax import random

from flax import errors
from flax.configurations import temp_flip_flag
from flax.core import Scope, apply, freeze, init, lazy_init, nn, scope
from flax.core.scope import LazyRng


class ScopeTest(absltest.TestCase):
  def test_rng(self):
    def f(scope):
      self.assertTrue(scope.has_rng('params'))
      self.assertFalse(scope.has_rng('dropout'))
      rng = scope.make_rng('params')
      self.assertTrue(
        np.all(rng == LazyRng.create(random.key(0), 1).as_jax_rng())
      )

    init(f)(random.key(0))

  def test_in_filter(self):
    filter_true = lambda x, y: self.assertTrue(scope.in_filter(x, y))
    filter_false = lambda x, y: self.assertFalse(scope.in_filter(x, y))

    filter_true(True, 'any_string1')
    filter_false(False, 'any_string2')
    filter_true('exact_match', 'exact_match')
    filter_false('no_match1', 'no_match2')
    filter_true(['one', 'two'], 'one')
    filter_false(['one', 'two'], 'three')
    filter_false([], 'one')
    filter_false([], None)

  def test_union_filter(self):
    def union_check(a, b, ans):
      self.assertEqual(scope.union_filters(a, b), ans)
      self.assertEqual(scope.union_filters(b, a), ans)

    union_check(['a', 'b'], ['b', 'c'], {'a', 'b', 'c'})
    union_check(True, False, True)
    union_check(False, False, set())
    union_check(True, True, True)
    union_check(
      scope.DenyList(['a', 'b']),
      scope.DenyList(['b', 'c']),
      scope.DenyList({'b'}),
    )
    union_check(
      scope.DenyList(['a', 'b']), ['b', 'c'], scope.DenyList({'a'})
    )

  def test_intersect_filter(self):
    def intersect_check(a, b, ans):
      self.assertEqual(scope.intersect_filters(a, b), ans)
      self.assertEqual(scope.intersect_filters(b, a), ans)

    intersect_check(['a', 'b'], ['b', 'c'], {'b'})
    intersect_check(True, False, False)
    intersect_check(False, False, set())
    intersect_check(True, True, True)
    intersect_check(
      scope.DenyList(['a', 'b']),
      scope.DenyList(['b', 'c']),
      scope.DenyList({'a', 'b', 'c'}),
    )
    intersect_check(scope.DenyList(['a', 'b']), ['b', 'c'], {'c'})

  def test_subtract_filter(self):
    def subtract_check(a, b, ans):
      self.assertEqual(scope.subtract_filters(a, b), ans)

    subtract_check(['a', 'b'], ['b', 'c'], {'a'})
    subtract_check(True, False, scope.DenyList(False))
    subtract_check(False, False, set())
    subtract_check(True, True, False)
    subtract_check(True, 'a', scope.DenyList('a'))
    subtract_check(
      scope.DenyList(['a', 'b']), scope.DenyList(['b', 'c']), {'c'}
    )
    subtract_check(
      scope.DenyList(['a', 'b']),
      ['b', 'c'],
      scope.DenyList({'a', 'b', 'c'}),
    )

  def test_group_collections(self):
    params = {'dense1': {'x': [10, 20]}}
    batch_stats = {'dense1': {'ema': 5}}
    xs = {'params': params, 'batch_stats': batch_stats}

    # Retrieve all keys only once.
    group = scope.group_collections(xs, ['params', 'params'])
    self.assertEqual(group, ({'params': params}, {}))

    # Ignore non-existing keys.
    self.assertEqual(scope.group_collections(xs, ['vars']), ({},))

    # False gets nothing and True retrieves all keys once.
    self.assertEqual(
      scope.group_collections(xs, [False, True, True]), ({}, xs, {})
    )

  def test_inconsistent_param_shapes(self):
    def f(scope):
      scope.param('test', nn.initializers.ones_init(), (4,))

    msg = (
        r'For parameter "test" in "/", the given initializer is expected to'
        r' generate shape \(4,\), but the existing parameter it received has'
        r' shape \(2,\).'
    )
    with self.assertRaisesRegex(errors.ScopeParamShapeError, msg):
      apply(f)(freeze({'params': {'test': np.ones((2,))}}))

  def test_apply_variables_bad_pytree(self):
    def f(scope):
      scope.param('kernel', nn.initializers.ones_init(), (4,))

    params = freeze(
      {
        'params': {
          'kernel': np.ones((4,)),
        },
      }
    )
    apply(f)(params)  # Valid.
    msg = 'but got a dict with an extra params layer'
    with self.assertRaisesRegex(
      errors.ApplyScopeInvalidVariablesStructureError, msg
    ):
      apply(f)({'params': params})

  def test_mutate_undefined_collection(self):
    def f(scope):
      scope.put_variable('state', 'test', 123)

    msg = (
      r'Cannot update variable "test" in "/" because collection "state" is'
      r' immutable.'
    )
    with self.assertRaisesRegex(errors.ModifyScopeVariableError, msg):
      init(f, mutable='params')(random.key(0))

  def test_undefined_param(self):
    def f(scope):
      nn.dense(scope.push('dense'), np.ones((1, 2)), 2)

    msg = r'Could not find parameter named "kernel" in scope "/dense".'
    with self.assertRaisesRegex(errors.ScopeParamNotFoundError, msg):
      apply(f)({'params': {'abc': 1}})

  def test_variable_is_mutable(self):
    def f(scope, should_be_mutable):
      test = scope.variable('state', 'test', lambda: 1)
      self.assertEqual(test.is_mutable(), should_be_mutable)

    _, variables = apply(f, mutable='state')({}, True)
    apply(f, mutable=False)(variables, False)

  def test_rngs_check_w_frozen_dict(self):
    def f(scope, x):
      return x

    _ = apply(f)({}, np.array([0.0]), rngs=freeze({'a': random.key(0)}))

  def test_rng_check_w_old_and_new_keys(self):
    # random.key always returns a new-style typed PRNG key.
    key = random.key(0)
    self.assertTrue(scope._is_valid_rng(key))
    self.assertFalse(scope._is_valid_rng(random.split(key)))

    # random.PRNGKey returns an old-style uint32 key by default.
    old_key = random.PRNGKey(0)
    self.assertTrue(scope._is_valid_rng(old_key))
    self.assertFalse(scope._is_valid_rng(random.split(old_key)))

    # Also explicitly test raw key data, because the jax_enable_custom_prng
    # flag can make PRNGKey return new-style keys.
    raw_key = random.key_data(key)
    self.assertTrue(scope._is_valid_rng(raw_key))
    self.assertFalse(scope._is_valid_rng(random.split(raw_key)))

  def test_rng_check_w_lazy_rng(self):
    key = random.key(0)
    self.assertTrue(scope._is_valid_rng(scope.LazyRng.create(key, 1)))

  def test_jax_leak_detector(self):
    with jax.check_tracer_leaks(True):

      def f(scope):
        def g(scope):
          pass

        scope.child(g)()

      jax.jit(init(f))(random.key(0))

  def test_rng_counter_reuse(self):
    root = Scope({}, {'dropout': random.key(0)})

    def f(scope):
      return scope.make_rng('dropout')

    a = root.child(f)()
    root = root.rewound()
    b = root.child(f)()
    self.assertFalse(jnp.allclose(a, b))

  def test_empty_col_error(self):
    root = Scope({})
    with self.assertRaises(errors.ScopeCollectionNotFound):
      root.param('test', nn.initializers.zeros_init(), ())
    root = Scope({'params': {}})
    with self.assertRaises(errors.ScopeCollectionNotFound):
      root.param('test', nn.initializers.zeros_init(), ())

    root = Scope({'params': {'abc': 1}})
    with self.assertRaises(errors.ScopeCollectionNotFound):
      root.variable('state', 'test', jnp.zeros, ())
    root = Scope({'state': {}})
    with self.assertRaises(errors.ScopeCollectionNotFound):
      root.variable('state', 'test', jnp.zeros, ())

  def test_variable_no_init(self):
    root = Scope({}, mutable='state')
    with self.assertRaises(errors.ScopeCollectionNotFound):
      root.variable('state', 'test')
    root = Scope({'state': {'abc': 1}}, mutable='state')
    abc = root.variable('state', 'abc')
    self.assertEqual(abc.value, 1)
    with self.assertRaises(errors.ScopeVariableNotFoundError):
      root.variable('state', 'test')

  def test_variable_alias(self):
    scope = Scope({}, mutable='state')
    subscope = scope.push(name='a')
    subscope.put_variable('state', 'x', 0.0)
    scope.put_variable('state', 'a', {'x': jnp.array(1.0, jnp.float32)})
    self.assertEqual(
      scope.variables()['state']['a']['x'], subscope.variables()['state']['x']
    )

  def test_lazy_init(self):
    def f(scope, x):
      k = scope.param(
        'kernel', nn.initializers.lecun_normal(), (x.shape[-1], x.shape[-1])
      )
      return x @ k

    init_fn = lazy_init(f)
    # provide a massive input message which would OOM if any compute ops were actually executed
    variables = init_fn(
      random.key(0),
      jax.ShapeDtypeStruct((1024 * 1024 * 1024, 128), jnp.float32),
    )
    self.assertEqual(variables['params']['kernel'].shape, (128, 128))

  def test_lazy_init_fails_on_data_dependence(self):
    def f(scope, x):
      # kernel is initialized with x so params are now dependent on the input
      k = scope.param('kernel', lambda _: x)
      return x * k

    init_fn = lazy_init(f)
    with self.assertRaises(errors.LazyInitError):
      init_fn(random.key(0), jax.ShapeDtypeStruct((8, 4), jnp.float32))

  @temp_flip_flag('fix_rng_separator', True)
  def test_fold_in_static_seperator(self):
    x = LazyRng(random.key(0), ('ab', 'c'))
    y = LazyRng(random.key(0), ('a', 'bc'))
    self.assertFalse(np.all(x.as_jax_rng() == y.as_jax_rng()))


if __name__ == '__main__':
  absltest.main()
