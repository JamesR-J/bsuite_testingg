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

from typing import Any, Callable, NamedTuple, Tuple

import distrax

from bsuite.baselines import base
from bsuite.baselines.utils import sequence

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


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: Any
    reward_state: Any


class TrainStateRP(TrainState):  # TODO check gradients do not update the static prior
    static_prior_params: flax.core.FrozenDict


class ActorCritic(base.Agent):
    """Feed-forward actor-critic agent."""

    def __init__(
            self,
            obs_spec: specs.Array,
            action_spec: specs.DiscreteArray,
            network: PolicyValueNet,
            optimizer: optax.GradientTransformation,
            rng: hk.PRNGSequence,
            sequence_length: int,
            discount: float,
            td_lambda: float,
    ):
        NUM_ENSEMBLE = 10
        RP_NOISE = 0.1
        LR = 1e-4

        def _get_reward_noise(ensrpr_state, obs, actions, key):
            SIGMA_SCALE = 3.0
            ensemble_obs = jnp.repeat(obs[jnp.newaxis, :], NUM_ENSEMBLE, axis=0)
            ensemble_action = jnp.repeat(actions[jnp.newaxis, :], NUM_ENSEMBLE, axis=0)

            def bootstrap_samples(key, action, obs, reward=None, m=10):
                n = action.shape[0]
                # Generate an array of shape (m, n) with random indices for bootstrapping
                indices = jrandom.randint(key, shape=(m, n), minval=0, maxval=n)
                # Use the indices to gather the bootstrapped samples
                bootstrapped_action = action[indices]
                bootstrapped_obs = obs[indices]
                if reward is not None:
                    bootstrapped_reward = reward[indices]
                else:
                    bootstrapped_reward = None
                return bootstrapped_action, bootstrapped_obs, bootstrapped_reward

            ensemble_action_bootstrap, ensemble_obs_bootstrap, _ = bootstrap_samples(key, actions, obs, m=NUM_ENSEMBLE)

            ensemble_obs = ensemble_obs_bootstrap
            ensemble_action = ensemble_action_bootstrap

            def single_reward_noise(ind_rpr_state, obs, action):
                action = jnp.expand_dims(action, axis=-1)
                rew_pred = ind_rpr_state.apply_fn({"params": {"static_prior": ind_rpr_state.static_prior_params,
                                                              "trainable": ind_rpr_state.params}},
                                                  (obs, action))
                return rew_pred

            ensembled_reward = jax.vmap(single_reward_noise)(ensrpr_state,
                                                             ensemble_obs,
                                                             ensemble_action)

            ensembled_reward = SIGMA_SCALE * jnp.std(ensembled_reward, axis=0)
            ensembled_reward = jnp.minimum(ensembled_reward, 1)

            return ensembled_reward

        def _reward_noise_over_actions(ensrpr_state: TrainStateRP, obs: chex.Array, key) -> chex.Array:
            # run the get_reward_noise for each action choice, can probs vmap
            actions = jnp.expand_dims(jnp.arange(0, action_spec.num_values, step=1), axis=-1)
            actions = jnp.tile(actions, obs.shape[0])

            obs = jnp.repeat(obs[jnp.newaxis, :], action_spec.num_values, axis=0)

            reward_over_actions = jax.vmap(_get_reward_noise, in_axes=(None, 0, 0, None))(ensrpr_state,
                                                                                         obs,
                                                                                         actions,
                                                                                         key)
            # reward_over_actions = jnp.sum(reward_over_actions, axis=0)  # TODO removed the layer sum
            reward_over_actions = jnp.swapaxes(jnp.squeeze(reward_over_actions, axis=-1), 0, 1)

            return reward_over_actions

        # Define loss function.
        def loss(trajectory: sequence.Trajectory, ensrpr_state, key) -> jnp.ndarray:
            """"Actor-critic loss."""
            logits, q_fn = network(trajectory.observations)
            action_probs = distrax.Softmax(logits=logits).probs
            values = action_probs * q_fn
            values = jnp.sum(values, axis=-1)

            values_tm1 = values[:-1]
            values_t = values[1:]

            _, actions, behaviour_logits, _, _, _ = jax.tree_util.tree_map(lambda t: t[:-1], trajectory)
            # TODO check the whole trajectory saving thing as is BAD
            learner_logits = logits[:-1]

            discount_t = trajectory.discounts[1:] * discount

            state_action_reward_noise = _get_reward_noise(ensrpr_state, trajectory.observations, trajectory.actions, key)
            state_reward_noise = _reward_noise_over_actions(ensrpr_state, trajectory.observations, key)

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

            total_loss = pg_loss + 0.5 * bl_loss + 0.01 * ent_loss
            return total_loss

        # Transform the loss into a pure function.
        loss_fn = hk.without_apply_rng(hk.transform(loss)).apply

        def uncertainty_loss(trajectory, ensrpr_state, key):
            def train_ensemble(indrpr_state, obs, actions, rewards):
                def reward_predictor_loss(rp_params, prior_params):
                    rew_pred = indrpr_state.apply_fn(
                        {"params": {"static_prior": prior_params, "trainable": rp_params}}, (obs, jnp.expand_dims(actions, axis=-1)))
                    return 0.5 * jnp.mean(jnp.square(rew_pred - rewards))

                ensemble_loss, grads = jax.value_and_grad(reward_predictor_loss, argnums=0)(indrpr_state.params,
                                                                                            indrpr_state.static_prior_params)
                indrpr_state = indrpr_state.apply_gradients(grads=grads)

                return ensemble_loss, indrpr_state

            key, _key = jrandom.split(key)
            obs = trajectory.observations
            action = trajectory.actions
            jitter_reward = trajectory.rewards + RP_NOISE * jrandom.normal(_key, shape=trajectory.rewards.shape)

            ensemble_state = jnp.repeat(obs[jnp.newaxis, :], NUM_ENSEMBLE, axis=0)
            ensemble_action = jnp.repeat(action[jnp.newaxis, :], NUM_ENSEMBLE, axis=0)
            ensemble_reward = jnp.repeat(jitter_reward[jnp.newaxis, :], NUM_ENSEMBLE, axis=0)

            def bootstrap_samples(key, action, obs, reward=None, m=10):
                n = action.shape[0]
                # Generate an array of shape (m, n) with random indices for bootstrapping
                indices = jrandom.randint(key, shape=(m, n), minval=0, maxval=n)
                # Use the indices to gather the bootstrapped samples
                bootstrapped_action = action[indices]
                bootstrapped_obs = obs[indices]
                if reward is not None:
                    bootstrapped_reward = reward[indices]
                else:
                    bootstrapped_reward = None
                return bootstrapped_action, bootstrapped_obs, bootstrapped_reward

            ensemble_action_bootstrap, ensemble_state_bootstrap, ensemble_reward_bootstrap \
                = bootstrap_samples(key, action, obs, jitter_reward, NUM_ENSEMBLE)

            ensemble_state = ensemble_state_bootstrap
            ensemble_action = ensemble_action_bootstrap
            ensemble_reward = ensemble_reward_bootstrap

            ensembled_loss, ensrpr_state = jax.vmap(train_ensemble)(ensrpr_state,
                                                                    ensemble_state,
                                                                    ensemble_action,
                                                                    ensemble_reward)

            return ensrpr_state, key

        # Define update function.
        @jax.jit
        def sgd_step(state: TrainingState,
                     trajectory: sequence.Trajectory,
                     key) -> TrainingState:
            """Does a step of SGD over a trajectory."""
            gradients = jax.grad(loss_fn)(state.params, trajectory, state.reward_state, key)
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

            ensrpr_state, key = uncertainty_loss(trajectory, state.reward_state, key)

            return TrainingState(params=new_params, opt_state=new_opt_state, reward_state=ensrpr_state), key

        # Initialize network parameters and optimiser state.
        init, forward = hk.without_apply_rng(hk.transform(network))
        dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)
        initial_params = init(next(rng), dummy_observation)
        initial_opt_state = optimizer.init(initial_params)

        def create_reward_state(key, rp_network):
            key, _key = jrandom.split(key)
            # rp_params = rp_network.init(_key,
            #                             (jnp.zeros((1, *obs_spec.shape)),
            #                              jnp.zeros((1, 1))))["params"]
            rp_params = rp_network.init(_key,
                                        (jrandom.uniform(_key, shape=(1, *obs_spec.shape), minval=0.0, maxval=0.5),
                                         jrandom.uniform(_key, shape=(1, 1), minval=0.0, maxval=0.5)))["params"]
            reward_state = TrainStateRP.create(apply_fn=rp_network.apply,
                                               params=rp_params["trainable"],
                                               static_prior_params=rp_params["static_prior"],
                                               tx=optax.adam(LR))
            return reward_state

        rp_network = RandomisedPrior()
        key = jrandom.PRNGKey(42)
        key, _key = jrandom.split(key)
        ensemble_keys = jrandom.split(_key, NUM_ENSEMBLE)
        ensembled_reward_state = jax.vmap(create_reward_state, in_axes=(0, None))(ensemble_keys, rp_network)

        # Internalize state.
        self._state = TrainingState(initial_params, initial_opt_state, ensembled_reward_state)
        self._forward = jax.jit(forward)
        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
        self._sgd_step = sgd_step
        self._rng = rng

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
        self._buffer.append(timestep, action, logits, new_timestep)
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

    return ActorCritic(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
        optimizer=optax.adam(3e-3),
        rng=hk.PRNGSequence(seed),
        sequence_length=32,
        discount=0.99,
        td_lambda=0.9,
    )


import flax.linen as nn


class PriorAndNotNN(nn.Module):

    @nn.compact
    def __call__(self, data):
        # takes in s and a
        obs, actions = data

        # obs = jnp.expand_dims(obs, axis=-1)
        # obs = nn.Conv(32, kernel_size=(2, 2), strides=(1, 1), padding="VALID")(obs)
        # obs = nn.relu(obs)
        # obs = nn.Conv(64, kernel_size=(2, 2), strides=(1, 1), padding="VALID")(obs)
        # obs = nn.relu(obs)

        obs = obs.reshape((obs.shape[0], -1))
        obs = nn.Dense(48)(obs)
        actions = nn.Dense(16)(actions)

        x = jnp.concatenate([obs, actions], axis=1)

        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        return x


class RandomisedPrior(nn.Module):
    static_prior: PriorAndNotNN = PriorAndNotNN()
    trainable: PriorAndNotNN = PriorAndNotNN()
    beta: float = 3

    @nn.compact
    def __call__(self, x):
        x1 = self.static_prior(x)
        x2 = self.trainable(x)

        return self.beta * x1 + x2
