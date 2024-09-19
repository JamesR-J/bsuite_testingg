# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple actor-critic agent implemented in JAX + Haiku."""

from typing import Any, Callable, NamedTuple, Tuple, Sequence

import distrax

from bsuite.baselines import base
from bsuite.baselines.utils import sequence

import numpy as np
import sys
import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
import jax.random as jrandom
import flax
from flax.training.train_state import TrainState
import chex
import jax
import jax.numpy as jnp
from rlax._src import distributions
from distrax._src.utils import math


Array = chex.Array
Logits = jnp.ndarray
Value = jnp.ndarray
PolicyValueNet = Callable[[jnp.ndarray], Tuple[Logits, Value]]


def entropy_loss_fn(logits_t, uncertainty_t, mask):
    log_pi = jax.nn.log_softmax(logits_t)
    log_pi_pi = math.mul_exp(log_pi, log_pi)
    entropy_per_timestep = -jnp.sum(log_pi_pi * uncertainty_t, axis=-1)
    # entropy_per_timestep = -jnp.sum(log_pi * uncertainty_t, axis=-1)
    return -jnp.mean(entropy_per_timestep * mask)


class EnsembleTrainingState(NamedTuple):
  params: hk.Params
  target_params: hk.Params
  opt_state: Any
  step: int


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: Any


class ActorCritic(base.Agent):
    """Feed-forward actor-critic agent."""

    def __init__(
            self,
            obs_spec: specs.Array,
            action_spec: specs.DiscreteArray,
            network: PolicyValueNet,
            ensemble_network: Any,
            optimizer: optax.GradientTransformation,
            ensemble_optimizer: optax.GradientTransformation,
            rng: hk.PRNGSequence,
            sequence_length: int,
            discount: float,
            td_lambda: float,
            noise_scale: float,
            num_ensemble: int
    ):
        def _get_reward_noise(obs, actions):
            ensemble_obs = jnp.repeat(obs[jnp.newaxis, :], num_ensemble, axis=0)
            ensemble_action = jnp.repeat(actions[jnp.newaxis, :], num_ensemble, axis=0)

            def single_reward_noise(state, obs, action):
                return ensemble_network.apply(state.params, obs, jnp.expand_dims(action, axis=-1))

            ensembled_reward = jnp.zeros((num_ensemble, 17, 1))  # TODO fix this 17
            for k, state in enumerate(self._ensemble):
                ensembled_reward.at[k].set(single_reward_noise(state, obs, actions))

            SIGMA_SCALE = 3.0
            ensembled_reward = SIGMA_SCALE * jnp.std(ensembled_reward, axis=0)
            ensembled_reward = jnp.minimum(ensembled_reward, 1)

            return ensembled_reward

        def _reward_noise_over_actions(obs: chex.Array) -> chex.Array:
            # run the get_reward_noise for each action choice, can probs vmap
            actions = jnp.expand_dims(jnp.arange(0, action_spec.num_values, step=1), axis=-1)
            actions = jnp.tile(actions, obs.shape[0])

            obs = jnp.repeat(obs[jnp.newaxis, :], action_spec.num_values, axis=0)

            reward_over_actions = jax.vmap(_get_reward_noise, in_axes=(0, 0))(obs, actions)
            # reward_over_actions = jnp.sum(reward_over_actions, axis=0)  # TODO removed the layer sum
            reward_over_actions = jnp.swapaxes(jnp.squeeze(reward_over_actions, axis=-1), 0, 1)

            return reward_over_actions

        # Define loss function.
        def loss(trajectory: sequence.Trajectory, key) -> jnp.ndarray:
            """"Actor-critic loss."""
            logits, q_fn = network(trajectory.observations)
            action_probs = distrax.Softmax(logits=logits).probs
            values = action_probs * q_fn
            values = jnp.sum(values, axis=-1)
            values_tm1 = values[:-1]
            values_t = values[1:]

            _, actions, behaviour_logits, _, _, _, _, _ = jax.tree_util.tree_map(lambda t: t[:-1], trajectory)
            # TODO check the whole trajectory saving thing as is BAD
            learner_logits = logits[:-1]

            discount_t = trajectory.discounts[1:] * discount

            state_action_reward_noise = _get_reward_noise(trajectory.observations, trajectory.actions)
            state_reward_noise = _reward_noise_over_actions(trajectory.observations)

            rewards = jnp.expand_dims(trajectory.rewards[1:], axis=-1) + state_action_reward_noise[1:]

            vtrace_td_error_and_advantage = jax.vmap(rlax.vtrace_td_error_and_advantage, in_axes=(1, 1, 1, 1, 1, None), out_axes=1)
            rhos = rlax.categorical_importance_sampling_ratios(learner_logits,
                                                               behaviour_logits,
                                                               actions)
            # TODO edit the vtrace policy returns stuff  https://lilianweng.github.io/posts/2018-04-08-policy-gradient/ inspo for the entropy bonus thing
            vtrace_returns = vtrace_td_error_and_advantage(jnp.expand_dims(values_tm1, axis=-1),
                                                           jnp.expand_dims(values_t, axis=-1),
                                                           rewards,
                                                           jnp.expand_dims(discount_t, axis=-1),
                                                           jnp.expand_dims(rhos, axis=-1),
                                                           0.9)

            mask = jnp.not_equal(trajectory.step[1:], int(dm_env.StepType.FIRST))
            mask = mask.astype(jnp.float32)

            pg_advantage = jax.lax.stop_gradient(vtrace_returns.pg_advantage)
            tb_pg_loss_fn = jax.vmap(rlax.policy_gradient_loss, in_axes=1, out_axes=0)
            pg_loss = tb_pg_loss_fn(jnp.expand_dims(learner_logits, axis=1), jnp.expand_dims(actions, axis=-1),
                                    pg_advantage, jnp.expand_dims(mask, axis=-1))
            pg_loss = jnp.mean(pg_loss)

            # pg_loss = jnp.mean(action_probs[:-1] * (-jax.nn.log_softmax(learner_logits) * state_reward_noise[:-1] - q_fn[:-1]))

            entropy_loss = jax.vmap(entropy_loss_fn, in_axes=1)(jnp.expand_dims(learner_logits, axis=1),
                                                                jnp.expand_dims(state_reward_noise[:-1], axis=1),
                                                                jnp.expand_dims(mask, axis=-1))
            ent_loss = jnp.mean(entropy_loss)

            # ent_loss_fn = jax.vmap(rlax.entropy_loss, in_axes=1, out_axes=0)
            # ent_loss = ent_loss_fn(jnp.expand_dims(learner_logits, axis=1), jnp.expand_dims(mask, axis=-1))
            # ent_loss = jnp.mean(ent_loss)

            # Baseline loss.
            bl_loss = 0.5 * jnp.mean(jnp.square(vtrace_returns.errors) * mask)

            # total_loss = pg_loss + 0.5 * bl_loss + 0.01 * ent_loss
            total_loss = pg_loss + 0.5 * bl_loss + ent_loss
            return total_loss

        # Transform the loss into a pure function.
        loss_fn = hk.without_apply_rng(hk.transform(loss)).apply

        # Transform the (impure) network into a pure function.
        ensemble_network = hk.without_apply_rng(hk.transform(ensemble_network))

        # Define loss function, including bootstrap mask `m_t` & reward noise `z_t`.
        def ensemble_loss(params: hk.Params,
                 transitions: Sequence[jnp.ndarray]) -> jnp.ndarray:
            """Q-learning loss with added reward noise + half-in bootstrap."""
            o_tm1, a_tm1, r_t, m_t, z_t = transitions
            r_t_pred = ensemble_network.apply(params, o_tm1, jnp.expand_dims(a_tm1, axis=-1))
            r_t += noise_scale * z_t
            return 0.5 * jnp.mean(m_t * jnp.square(r_t - r_t_pred))  # TODO is this right?

        # Define update function.
        @jax.jit
        def sgd_step(state: TrainingState,
                     trajectory: sequence.Trajectory,
                     key) -> TrainingState:
            """Does a step of SGD over a trajectory."""
            gradients = jax.grad(loss_fn)(state.params, trajectory, key)
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

            # ensrpr_state, key = uncertainty_loss(trajectory, state.reward_state, key)
            for k, ensemble_state in enumerate(self._ensemble):
                transitions = [trajectory.observations, trajectory.actions, trajectory.rewards, trajectory.mask[:, k], trajectory.noise[:, k]]
                self._ensemble[k] = ensemble_sgd_step(ensemble_state, transitions)

            return TrainingState(params=new_params, opt_state=new_opt_state), key

        # Define update function for each member of ensemble..
        @jax.jit
        def ensemble_sgd_step(state: EnsembleTrainingState,
                     transitions: Sequence[jnp.ndarray]) -> EnsembleTrainingState:
            """Does a step of SGD for the whole ensemble over `transitions`."""

            gradients = jax.grad(ensemble_loss)(state.params, transitions)
            updates, new_opt_state = ensemble_optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

            # print(new_params)

            return EnsembleTrainingState(
                params=new_params,
                target_params=state.target_params,
                opt_state=new_opt_state,
                step=state.step + 1)

        # Initialize network parameters and optimiser state.
        init, forward = hk.without_apply_rng(hk.transform(network))
        dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, 1), dtype=jnp.int32)
        initial_params = init(next(rng), dummy_observation)
        initial_opt_state = optimizer.init(initial_params)

        initial_ensemble_params = [
            ensemble_network.init(next(rng), dummy_observation, dummy_action) for _ in range(num_ensemble)
        ]
        initial_ensemble_target_params = [
            ensemble_network.init(next(rng), dummy_observation, dummy_action) for _ in range(num_ensemble)
        ]
        initial_ensemble_opt_state = [ensemble_optimizer.init(p) for p in initial_ensemble_params]

        # Internalize state.
        self._state = TrainingState(initial_params, initial_opt_state)
        self._ensemble = [
            EnsembleTrainingState(p, tp, o, step=0) for p, tp, o in zip(
                initial_ensemble_params, initial_ensemble_target_params, initial_ensemble_opt_state)
        ]
        self._forward = jax.jit(forward)
        self._ensemble_forward = jax.jit(ensemble_network.apply)
        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length, num_ensemble)
        self._sgd_step = sgd_step
        self._rng = rng
        self._num_ensemble = num_ensemble
        self._mask_prob = 1.0

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to a softmax policy."""
        key = next(self._rng)
        observation = timestep.observation[None, ...]
        logits, _ = self._forward(self._state.params, observation)
        action = jax.random.categorical(key, logits).squeeze()
        return int(action), logits

    def update(
            self,
            timestep: dm_env.TimeStep,
            action: base.Action,
            logits,
            new_timestep: dm_env.TimeStep,
            key
    ):
        """Adds a transition to the trajectory buffer and periodically does SGD."""
        mask = np.random.binomial(1, self._mask_prob, self._num_ensemble)
        noise = np.random.randn(self._num_ensemble)

        self._buffer.append(timestep, action, logits, new_timestep, mask, noise)
        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()
            self._state, key = self._sgd_step(self._state, trajectory, key)

        return key


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  seed: int = 0) -> base.Agent:
    """Creates an actor-critic agent with default hyperparameters."""

    def network(inputs: jnp.ndarray) -> Tuple[Logits, Value]:
        flat_inputs = hk.Flatten()(inputs)
        torso = hk.nets.MLP([64, 64])
        policy_head = hk.Linear(action_spec.num_values)
        value_head = hk.Linear(action_spec.num_values)
        # value_head = hk.Linear(1)
        embedding = torso(flat_inputs)
        logits = policy_head(embedding)
        value = value_head(embedding)
        return logits, value  #  jnp.squeeze(value, axis=-1)

    prior_scale = 5.
    hidden_sizes = [50, 50]

    def ensemble_network(obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Simple Q-network with randomized prior function."""
        net = hk.nets.MLP([*hidden_sizes, 1])
        prior_net = hk.nets.MLP([*hidden_sizes, 1])
        obs = hk.Flatten()(obs)
        obs = hk.Linear(49)(obs)
        # actions = hk.Linear(25)(actions)
        x = jnp.concatenate((obs, actions), axis=-1)
        return net(x) + prior_scale * jax.lax.stop_gradient(prior_net(x))

    return ActorCritic(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
        ensemble_network=ensemble_network,
        optimizer=optax.adam(3e-3),
        ensemble_optimizer=optax.adam(1e-4),
        rng=hk.PRNGSequence(seed),
        sequence_length=32,
        discount=0.99,
        td_lambda=0.9,
        noise_scale=0.1,
        num_ensemble=10
    )
