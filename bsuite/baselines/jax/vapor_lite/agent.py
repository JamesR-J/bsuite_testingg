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
import flashbax

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


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    logits: jnp.ndarray
    rewards: jnp.ndarray
    discounts: jnp.ndarray
    step: jnp.ndarray
    mask: jnp.ndarray
    noise: jnp.ndarray


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
            mask_prob: float,
            num_ensemble: int,
            importance_sampling_exponent: int
    ):
        def _get_reward_noise(obs, actions):
            def single_reward_noise(state, obs, action):
                reward_pred = ensemble_network.apply(state.params, obs, jnp.expand_dims(action, axis=-1))
                return reward_pred

            ensembled_reward_sep = jnp.zeros((num_ensemble, actions.shape[0], 1))
            for k, state in enumerate(self._ensemble):
                ensembled_reward_sep = ensembled_reward_sep.at[k].set(single_reward_noise(state, obs, actions))

            SIGMA_SCALE = 3.0  # this are from other experiments
            ensembled_reward = SIGMA_SCALE * jnp.std(ensembled_reward_sep, axis=0)
            # ensembled_reward = jnp.var(ensembled_reward_sep, axis=0)
            ensembled_reward = jnp.minimum(ensembled_reward, 1)

            return ensembled_reward, ensembled_reward_sep

        def _reward_noise_over_actions(obs: chex.Array) -> chex.Array:
            # run the get_reward_noise for each action choice, can probs vmap
            actions = jnp.expand_dims(jnp.arange(0, action_spec.num_values, step=1), axis=-1)
            actions = jnp.tile(actions, obs.shape[0])
            actions = jnp.expand_dims(actions, axis=-1)

            obs = jnp.repeat(obs[jnp.newaxis, :], action_spec.num_values, axis=0)

            reward_over_actions = jax.vmap(_get_reward_noise, in_axes=(0, 0))(obs, actions)
            reward_over_actions = jnp.swapaxes(jnp.squeeze(reward_over_actions, axis=-1), 0, 1)

            return reward_over_actions

        # Define loss function.
        def loss(trajectory) -> jnp.ndarray:
            """"Actor-critic loss."""
            logits, values = network(trajectory.observations)
            policy_dist = distrax.Softmax(logits=logits[:-1])
            log_prob = policy_dist.log_prob(trajectory.actions)

            state_action_reward_noise, ensembled_reward_sep = _get_reward_noise(trajectory.observations[:-1], trajectory.actions)
            state_reward_noise = _get_reward_noise(trajectory.observations, trajectory.actions)

            td_lambda = jax.vmap(rlax.td_lambda, in_axes=(1, 1, 1, 1, None), out_axes=1)
            k_estimate = td_lambda(jnp.expand_dims(values[:-1], axis=-1),
                                   jnp.expand_dims(trajectory.rewards, axis=-1) + state_action_reward_noise,
                                   jnp.expand_dims(trajectory.discounts * discount, axis=-1),
                                   jnp.expand_dims(values[1:], axis=-1),
                                   jnp.array(td_lambda_val),
                                   )

            value_loss = jnp.mean(jnp.square(values[:-1] - jax.lax.stop_gradient(k_estimate - tau * log_prob)))
            # TODO is it right to use [1:] for these values etc or [:-1]?

            entropy = policy_dist.entropy()

            entropy_loss = jax.vmap(entropy_loss_fn, in_axes=1)(jnp.expand_dims(learner_logits, axis=1),
                                                                jnp.expand_dims(state_reward_noise[:-1], axis=1),
                                                                mask)
            ent_loss = jnp.mean(entropy_loss)

            # ent_loss_fn = jax.vmap(rlax.entropy_loss, in_axes=1, out_axes=0)
            # ent_loss = ent_loss_fn(jnp.expand_dims(learner_logits, axis=1), jnp.expand_dims(mask, axis=-1))
            # ent_loss = jnp.mean(ent_loss)

            policy_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(k_estimate - values[:-1]) - tau * entropy)
            # TODO unsure if above correct, true to pseudo code but I think the log_prob should also multiple the entropy potentially

            return policy_loss + value_loss, (entropy, jax.lax.stop_gradient(ensembled_reward_sep))

        # Transform the loss into a pure function.
        critic_loss_fn = hk.without_apply_rng(hk.transform(critic_loss)).apply
        actor_loss_fn = hk.without_apply_rng(hk.transform(actor_loss)).apply

        # Transform the (impure) network into a pure function.
        ensemble_network = hk.without_apply_rng(hk.transform(ensemble_network))

        # Define loss function, including bootstrap mask `m_t` & reward noise `z_t`.
        def ensemble_loss(params: hk.Params,
                          transitions: Sequence[jnp.ndarray]) -> jnp.ndarray:
            """Q-learning loss with added reward noise + half-in bootstrap."""
            o_tm1, a_tm1, r_t, m_t, z_t = transitions
            r_t_pred = ensemble_network.apply(params, o_tm1, a_tm1)
            r_t += noise_scale * z_t
            return 0.5 * jnp.mean(m_t * jnp.square(r_t - r_t_pred))  # TODO is this right?

        # Define update function.
        @jax.jit
        def sgd_step(critic_state: TrainingState,
                     actor_state: TrainingState,
                     buffer_state,
                     trajectory: sequence.Trajectory,
                     key) -> TrainingState:
            """Does a step of SGD over a trajectory."""
            logits = self._actor_forward(self._actor_state.params, trajectory.experience["obs"][0])
            # TODO above dodgy but cba to fix
            gradients, (pg_advantage, new_priorities) = jax.grad(critic_loss_fn, has_aux=True)(critic_state.params, trajectory, logits, key)
            updates, new_critic_opt_state = optimizer.update(gradients, critic_state.opt_state)
            new_critic_params = optax.apply_updates(critic_state.params, updates)


            # TODO update priorities
            # new_priorities =
            #
            # buffer_state = self._fbx_buffer.set_priorities(buffer_state, trajectory.indices, new_priorities)

            gradients = jax.grad(actor_loss_fn)(actor_state.params, trajectory, pg_advantage, key)
            updates, new_actor_opt_state = optimizer.update(gradients, actor_state.opt_state)
            new_actor_params = optax.apply_updates(actor_state.params, updates)

            # ensrpr_state, key = uncertainty_loss(trajectory, state.reward_state, key)
            for k, ensemble_state in enumerate(self._ensemble):
                transitions = [trajectory.experience["obs"][0],
                               trajectory.experience["actions"][0],
                               trajectory.experience["rewards"][0],
                               trajectory.experience["mask"][0, :, k],
                               trajectory.experience["noise"][0, :, k]]
                self._ensemble[k] = ensemble_sgd_step(ensemble_state, transitions)

            return (TrainingState(params=new_critic_params, opt_state=new_critic_opt_state),
                    TrainingState(params=new_actor_params, opt_state=new_actor_opt_state),
                    buffer_state,
                    key)

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
        critic_init, critic_forward = hk.without_apply_rng(hk.transform(critic_network))
        actor_init, actor_forward = hk.without_apply_rng(hk.transform(actor_network))
        dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, 1), dtype=jnp.int32)
        initial_critic_params = critic_init(next(rng), dummy_observation)
        initial_actor_params = critic_init(next(rng), dummy_observation)
        initial_opt_state = optimizer.init(initial_critic_params)

        sample_seq_length = obs_spec.shape[0]  # TODO needs to be size of env
        self._fbx_buffer = flashbax.make_prioritised_trajectory_buffer(add_batch_size=2,
                                                                       sample_batch_size=2,
                                                                       sample_sequence_length=sample_seq_length+1,
                                                                       period=sample_seq_length+1, # So no overlap in trajs?
                                                                       min_length_time_axis=1,
                                                                       max_size=100,
                                                                       priority_exponent=0.6
                                                                       )

        initial_ensemble_params = [
            ensemble_network.init(next(rng), dummy_observation, dummy_action) for _ in range(num_ensemble)
        ]
        initial_ensemble_target_params = [
            ensemble_network.init(next(rng), dummy_observation, dummy_action) for _ in range(num_ensemble)
        ]
        initial_ensemble_opt_state = [ensemble_optimizer.init(p) for p in initial_ensemble_params]

        # Internalize state.
        self._critic_state = TrainingState(initial_critic_params, initial_opt_state)
        self._actor_state = TrainingState(initial_actor_params, initial_opt_state)
        self._ensemble = [
            EnsembleTrainingState(p, tp, o, step=0) for p, tp, o in zip(
                initial_ensemble_params, initial_ensemble_target_params, initial_ensemble_opt_state)
        ]
        self._critic_forward = jax.jit(critic_forward)
        self._actor_forward = jax.jit(actor_forward)
        self._ensemble_forward = jax.jit(ensemble_network.apply)
        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length, num_ensemble)
        self._sgd_step = sgd_step
        self._rng = rng
        self._num_ensemble = num_ensemble
        self._mask_prob = mask_prob
        self._action_spec = action_spec
        self._obs_spec = obs_spec
        self._importance_sampling_exponent = importance_sampling_exponent

    def return_buffer(self):
        fake_timestep = {"obs": jnp.zeros((*self._obs_spec.shape,)),
                         "actions": jnp.zeros((1,), dtype=self._action_spec.dtype),
                         "logits": jnp.zeros((self._action_spec.num_values,)),
                         "rewards": jnp.zeros((1,)),
                         "discounts": jnp.zeros((1,)),
                         "step": jnp.zeros((1,)),
                         "mask": jnp.zeros((self._num_ensemble,)),
                         "noise": jnp.zeros((self._num_ensemble,)),
                         }
        return self._fbx_buffer.init(fake_timestep)

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to a softmax policy."""
        key = next(self._rng)
        observation = timestep.observation[None, ...]
        logits = self._actor_forward(self._actor_state.params, observation)
        action = jax.random.categorical(key, logits).squeeze()
        return int(action), logits

    def update(
            self,
            timestep: dm_env.TimeStep,
            action: base.Action,
            logits,
            new_timestep: dm_env.TimeStep,
            buffer_state,
            key
    ):
        """Adds a transition to the trajectory buffer and periodically does SGD."""
        mask = np.random.binomial(1, self._mask_prob, self._num_ensemble)
        noise = np.random.randn(self._num_ensemble)

        self._buffer.append(timestep, action, logits, new_timestep, mask, noise)

        # self._buffer.append(timestep, action, logits, new_timestep, mask, noise)  # TODO ignore this dodgyness for now
        # if self._buffer.full() or new_timestep.last():
        #     trajectory = self._buffer.drain()
        #     self._state, key = self._sgd_step(self._state, trajectory, key)
        if new_timestep.last():
            trajectory = self._buffer.drain()
            buffer_data = {"obs": trajectory.observations,
                           "actions": jnp.expand_dims(trajectory.actions, axis=-1),
                           "logits": trajectory.logits,
                           "rewards": jnp.expand_dims(trajectory.rewards, axis=-1),
                           "discounts": jnp.expand_dims(trajectory.discounts, axis=-1),
                           "step": jnp.expand_dims(trajectory.step, axis=-1),
                           "mask": trajectory.mask,
                           "noise": trajectory.noise,
                           }
            broadcast_fn = lambda x: jnp.broadcast_to(x, (2, *x.shape))  # add batch dim
            fake_batch_sequence = jax.tree_util.tree_map(broadcast_fn, buffer_data)
            buffer_state = self._fbx_buffer.add(buffer_state,
                                                fake_batch_sequence
                                                )
            key, _key = jrandom.split(key)
            batch = self._fbx_buffer.sample(buffer_state, _key)
            self._critic_state, self._actor_state, buffer_state, key = self._sgd_step(self._critic_state,
                                                                                      self._actor_state,
                                                                                      buffer_state,
                                                                                      batch,
                                                                                      key)

        return buffer_state, key


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  config,
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

    prior_scale = config.PRIOR_SCALE
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
        optimizer=optax.adam(config.LR),
        ensemble_optimizer=optax.adam(config.ENS_LR),
        rng=hk.PRNGSequence(seed),
        sequence_length=50,
        discount=config.GAMMA,
        td_lambda_val=config.TD_LAMBDA,
        reward_noise_scale=config.REWARD_NOISE_SCALE,
        mask_prob=config.MASK_PROB,
        num_ensemble=10,
        init_tau=0.001
        importance_sampling_exponent=0.995
    )
