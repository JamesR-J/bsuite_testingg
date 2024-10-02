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
    tau: float


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
            td_lambda_val: float,
            noise_scale: float,
            init_tau: float,
            num_ensemble: int
    ):
        def _get_reward_noise(obs, actions):
            ensemble_obs = jnp.repeat(obs[jnp.newaxis, :], num_ensemble, axis=0)
            ensemble_action = jnp.repeat(actions[jnp.newaxis, :], num_ensemble, axis=0)
            # TODO check ye old ensemble obs and action and why not using them

            def single_reward_noise(state, obs, action):
                return ensemble_network.apply(state.params, obs, jnp.expand_dims(action, axis=-1))

            ensembled_reward = jnp.zeros((num_ensemble, self._obs_spec.shape[0], 1))  # TODO change to sequence length if not emptying buffer at end of episode
            # ensembled_reward = jnp.zeros((num_ensemble, sequence_length, 1))
            for k, state in enumerate(self._ensemble):
                ensembled_reward = ensembled_reward.at[k].set(single_reward_noise(state, obs, actions))

            SIGMA_SCALE = 3.0
            ensembled_reward = SIGMA_SCALE * jnp.std(ensembled_reward, axis=0)
            ensembled_reward = jnp.minimum(ensembled_reward, 1)

            return ensembled_reward

        # Define loss function.
        def loss(trajectory: sequence.Trajectory, tau) -> jnp.ndarray:
            """"Actor-critic loss."""
            logits, values = network(trajectory.observations)
            # policy_dist = distrax.Softmax(logits=logits[:-1])
            # log_prob = policy_dist.log_prob(trajectory.actions)
            #
            # state_action_reward_noise = _get_reward_noise(trajectory.observations[:-1], trajectory.actions)
            #
            # td_lambda = jax.vmap(rlax.td_lambda, in_axes=(1, 1, 1, 1, None), out_axes=1)
            # k_estimate = td_lambda(jnp.expand_dims(values[:-1], axis=-1),
            #     jnp.expand_dims(trajectory.rewards, axis=-1),  # + (state_action_reward_noise ** 2 / 2 * tau),
            #     jnp.expand_dims(trajectory.discounts * discount, axis=-1),
            #     jnp.expand_dims(values[1:], axis=-1),
            #     jnp.array(td_lambda_val),
            # )
            #
            # value_loss = jnp.mean(jnp.square(values[:-1] - jax.lax.stop_gradient(k_estimate - tau * log_prob)))
            # # TODO is it right to use [1:] for these values etc or [:-1]?
            #
            # entropy = -log_prob
            # # entropy = policy_dist.entropy()
            #
            # policy_loss = jnp.mean(log_prob * jax.lax.stop_gradient(k_estimate - values[:-1] - tau * entropy))

            td_errors = rlax.td_lambda(
                v_tm1=values[:-1],
                r_t=trajectory.rewards,
                discount_t=trajectory.discounts * discount,
                v_t=values[1:],
                lambda_=jnp.array(td_lambda_val),
            )
            value_loss = jnp.mean(td_errors ** 2)
            policy_loss = rlax.policy_gradient_loss(
                logits_t=logits[:-1],
                a_t=trajectory.actions,
                adv_t=td_errors,
                w_t=jnp.ones_like(td_errors))
            entropy = jnp.zeros(())

            return policy_loss + value_loss, entropy

        def tau_loss(tau, trajectory: sequence.Trajectory, entropy, key) -> jnp.ndarray:
            state_action_reward_noise = _get_reward_noise(trajectory.observations[:-1], trajectory.actions)

            return jnp.mean((state_action_reward_noise ** 2 / (2 * tau)) + tau * entropy)

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
            # r_t += noise_scale * z_t  # TODO don't need this when on policy
            return 0.5 * jnp.mean(m_t * jnp.square(r_t - r_t_pred))  # TODO is this right?

        # Define update function.
        @jax.jit
        def sgd_step(state: TrainingState,
                     trajectory: sequence.Trajectory) -> TrainingState:
            """Does a step of SGD over a trajectory."""
            gradients, entropy = jax.grad(loss_fn, has_aux=True)(state.params, trajectory, state.tau)
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

            # tau_gradients = jax.grad(tau_loss)(state.tau, trajectory, entropy, key)
            # tau = state.tau - tau_gradients  # TODO is this okay?
            tau = state.tau  # TODO fixed tau for now to figure out the algo innit

            # def callback(tau):
            #     print(tau)
            # jax.experimental.io_callback(callback, None, tau)

            # for k, ensemble_state in enumerate(self._ensemble):
            #     transitions = [trajectory.observations[:-1], trajectory.actions, trajectory.rewards,
            #                    trajectory.mask[:, k], trajectory.noise[:, k]]
            #     # TODO is this right observations [:-1]
            #     self._ensemble[k] = ensemble_sgd_step(ensemble_state, transitions)

            return TrainingState(params=new_params, opt_state=new_opt_state, tau=tau)

        # Define update function for each member of ensemble.
        @jax.jit
        def ensemble_sgd_step(state: EnsembleTrainingState,
                     transitions: Sequence[jnp.ndarray]) -> EnsembleTrainingState:
            """Does a step of SGD for the whole ensemble over `transitions`."""

            gradients = jax.grad(ensemble_loss)(state.params, transitions)
            updates, new_opt_state = ensemble_optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

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
        self._state = TrainingState(initial_params, initial_opt_state, init_tau)
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
        self._obs_spec = obs_spec

    def return_buffer(self):
        return None

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to a softmax policy."""
        key = next(self._rng)
        observation = timestep.observation[None, ...]
        logits, _ = self._forward(self._state.params, observation)
        action = jax.random.categorical(key, logits).squeeze()
        return int(action), logits

    def update(self,
            timestep: dm_env.TimeStep,
            action: base.Action,
            logits,
            new_timestep: dm_env.TimeStep,
               buffer_state
    ):
        """Adds a transition to the trajectory buffer and periodically does SGD."""
        mask = np.random.binomial(1, self._mask_prob, self._num_ensemble)
        noise = np.random.randn(self._num_ensemble)

        self._buffer.append(timestep, action, logits, new_timestep, mask, noise)

        # if self._buffer.full() or new_timestep.last():
        if self._buffer.full():
            trajectory = self._buffer.drain()
            self._state = self._sgd_step(self._state, trajectory)

        return buffer_state


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  seed: int = 0) -> base.Agent:
    """Creates an actor-critic agent with default hyperparameters."""

    def network(inputs: jnp.ndarray) -> Tuple[Logits, Value]:
        flat_inputs = hk.Flatten()(inputs)
        torso = hk.nets.MLP([64, 64])
        policy_head = hk.Linear(action_spec.num_values)
        # value_head = hk.Linear(action_spec.num_values)
        value_head = hk.Linear(1)
        embedding = torso(flat_inputs)
        logits = policy_head(embedding)
        value = value_head(embedding)
        # return logits, value  #  jnp.squeeze(value, axis=-1)
        return logits, jnp.squeeze(value, axis=-1)

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
        ensemble_optimizer=optax.adam(3e-3),
        rng=hk.PRNGSequence(seed),
        sequence_length=50,
        discount=0.99,
        td_lambda_val=0.9,
        noise_scale=0.0,
        num_ensemble=10,
        init_tau=0.001
    )
