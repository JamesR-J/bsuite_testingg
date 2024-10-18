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
from functools import partial
import wandb

Array = chex.Array
Logits = jnp.ndarray
Value = jnp.ndarray
PolicyValueNet = Callable[[jnp.ndarray], Tuple[Logits, Value]]


def entropy_loss_fn(logits_t, uncertainty_t, mask):
    log_pi = jax.nn.log_softmax(logits_t)
    log_pi_pi = math.mul_exp(log_pi, log_pi)
    # entropy_per_timestep = -jnp.sum(log_pi_pi * uncertainty_t, axis=-1)  # sigma log_pi pi
    entropy_per_timestep = -jnp.sum(log_pi * uncertainty_t, axis=-1)  # sigma log_pi
    return -jnp.mean(entropy_per_timestep * mask)


class EnsembleTrainingState(NamedTuple):
    params: hk.Params
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
            reward_noise_scale: float,
            mask_prob: float,
            num_ensemble: int,
            importance_sampling_exponent: int
    ):
        # Define loss function.
        def loss(trajectory, state_action_reward_noise, state_reward_noise) -> jnp.ndarray:
            """"Actor-critic loss."""
            logits, values = network(trajectory.observations)
            policy_dist = distrax.Softmax(logits=logits[:-1])
            log_prob = policy_dist.log_prob(trajectory.actions)
            action_probs = policy_dist.probs

            td_lambda = jax.vmap(rlax.td_lambda, in_axes=(1, 1, 1, 1, None), out_axes=1)
            q_estimate = td_lambda(jnp.expand_dims(values[:-1], axis=-1),
                                   jnp.expand_dims(trajectory.rewards, axis=-1) + state_action_reward_noise,
                                   jnp.expand_dims(trajectory.discounts * discount, axis=-1),
                                   jnp.expand_dims(values[1:], axis=-1),
                                   jnp.array(td_lambda_val),
                                   )

            value_loss = jnp.mean(jnp.square(values[:-1] - jax.lax.stop_gradient(q_estimate)))

            mask = jnp.not_equal(trajectory.step, int(dm_env.StepType.FIRST))
            mask = mask.astype(jnp.float32)
            entropy_loss = jax.vmap(entropy_loss_fn, in_axes=1)(jnp.expand_dims(logits[:-1], axis=1),
                                                                jnp.expand_dims(state_reward_noise, axis=1),
                                                                jnp.expand_dims(mask, axis=-1))
            entropy = jnp.mean(entropy_loss)

            # ent_loss_fn = jax.vmap(rlax.entropy_loss, in_axes=1, out_axes=0)
            # ent_loss = ent_loss_fn(jnp.expand_dims(learner_logits, axis=1), jnp.expand_dims(mask, axis=-1))
            # ent_loss = jnp.mean(ent_loss)

            # policy_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(q_estimate - values[:-1]) - entropy)
            #
            policy_loss = jnp.mean(action_probs * entropy - q_estimate)

            return policy_loss + value_loss

        # Define loss function, including bootstrap mask `m_t` & reward noise `z_t`.
        def ensemble_loss(params: hk.Params,
                          transitions: Sequence[jnp.ndarray]) -> jnp.ndarray:
            """Q-learning loss with added reward noise + half-in bootstrap."""
            o_tm1, a_tm1, r_t, m_t, z_t = transitions
            r_t_pred = ensemble_network.apply(params, o_tm1, jnp.expand_dims(a_tm1, axis=-1))
            # r_t += reward_noise_scale * z_t  # TODO don't need this when on policy
            loss = 0.5 * jnp.mean(m_t * jnp.square(r_t - jnp.squeeze(r_t_pred, axis=-1)))  # TODO is this right?

            return loss

        # Transform the loss into a pure function.
        loss_fn = hk.without_apply_rng(hk.transform(loss)).apply

        # Transform the (impure) network into a pure function.
        ensemble_network = hk.without_apply_rng(hk.transform(ensemble_network))

        # Define update function for each member of ensemble..
        @jax.jit
        def ensemble_sgd_step(state: EnsembleTrainingState,
                              transitions: Sequence[jnp.ndarray]) -> Tuple[EnsembleTrainingState, Any]:
            """Does a step of SGD for the whole ensemble over `transitions`."""

            ensemble_loss_val, gradients = jax.value_and_grad(ensemble_loss)(state.params, transitions)
            updates, new_opt_state = ensemble_optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

            return EnsembleTrainingState(params=new_params,
                                         opt_state=new_opt_state,
                                         step=state.step + 1), ensemble_loss_val

        # Define update function.
        @jax.jit
        def sgd_step(state: TrainingState,
                     buffer_state,
                     trajectory: sequence.Trajectory,
                     state_action_reward_noise,
                     state_reward_noise) -> TrainingState:
            """Does a step of SGD over a trajectory."""
            pv_loss, gradients = jax.value_and_grad(loss_fn, has_aux=False)(state.params, trajectory,
                                                                            state_action_reward_noise,
                                                                            state_reward_noise)
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)


            # TODO update priorities
            # new_priorities =
            #
            # buffer_state = self._fbx_buffer.set_priorities(buffer_state, trajectory.indices, new_priorities)

            return (TrainingState(params=new_params, opt_state=new_opt_state),
                    pv_loss,
                    buffer_state)



        # Initialize network parameters and optimiser state.
        init, forward = hk.without_apply_rng(hk.transform(network))
        dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, 1), dtype=jnp.int32)
        initial_params = init(next(rng), dummy_observation)
        initial_opt_state = optimizer.init(initial_params)

        # sample_seq_length = obs_spec.shape[0]  # TODO needs to be size of env
        # self._fbx_buffer = flashbax.make_prioritised_trajectory_buffer(add_batch_size=2,
        #                                                                sample_batch_size=2,
        #                                                                sample_sequence_length=sample_seq_length+1,
        #                                                                period=sample_seq_length+1, # So no overlap in trajs?
        #                                                                min_length_time_axis=1,
        #                                                                max_size=100,
        #                                                                priority_exponent=0.6
        #                                                                )

        initial_ensemble_params = [
            ensemble_network.init(next(rng), dummy_observation, dummy_action) for _ in range(num_ensemble)
        ]
        initial_ensemble_opt_state = [ensemble_optimizer.init(p) for p in initial_ensemble_params]

        # Internalize state.
        self._state = TrainingState(initial_params, initial_opt_state)
        self._ensemble = [
            EnsembleTrainingState(p, o, step=0) for p, o in zip(
                initial_ensemble_params, initial_ensemble_opt_state)
        ]
        self._forward = jax.jit(forward)
        self._ensemble_forward = jax.jit(ensemble_network.apply)
        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length, num_ensemble)
        self._sgd_step = sgd_step
        self._ensemble_sgd_step = ensemble_sgd_step
        self._rng = rng
        self._num_ensemble = num_ensemble
        self._mask_prob = mask_prob
        self._action_spec = action_spec
        self._obs_spec = obs_spec
        self._importance_sampling_exponent = importance_sampling_exponent

    def return_buffer(self):
        # fake_timestep = {"obs": jnp.zeros((*self._obs_spec.shape,)),
        #                  "actions": jnp.zeros((1,), dtype=self._action_spec.dtype),
        #                  "logits": jnp.zeros((self._action_spec.num_values,)),
        #                  "rewards": jnp.zeros((1,)),
        #                  "discounts": jnp.zeros((1,)),
        #                  "step": jnp.zeros((1,)),
        #                  "mask": jnp.zeros((self._num_ensemble,)),
        #                  "noise": jnp.zeros((self._num_ensemble,)),
        #                  }
        # return self._fbx_buffer.init(fake_timestep)
        return None

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions according to a softmax policy."""
        key = next(self._rng)
        observation = timestep.observation[None, ...]
        logits, _ = self._forward(self._state.params, observation)
        action = jax.random.categorical(key, logits).squeeze()
        return int(action), logits

    @partial(jax.jit, static_argnums=(0,))
    def _single_reward_noise(self, state, obs, action):
        reward_pred = self._ensemble_forward(state.params, obs, jnp.expand_dims(action, axis=-1))
        return reward_pred

    def _get_reward_noise(self, obs, actions):
        ensembled_reward_sep = jnp.zeros((self._num_ensemble, actions.shape[0], 1))
        for k, state in enumerate(self._ensemble):
            ensembled_reward_sep = ensembled_reward_sep.at[k].set(self._single_reward_noise(state, obs, actions))

        SIGMA_SCALE = 3.0  # this are from other experiments
        ensembled_reward = SIGMA_SCALE * jnp.std(ensembled_reward_sep, axis=0)
        # ensembled_reward = jnp.var(ensembled_reward_sep, axis=0)
        ensembled_reward = jnp.minimum(ensembled_reward, 1)

        return ensembled_reward, ensembled_reward_sep

    def _reward_noise_over_actions(self, obs: chex.Array) -> chex.Array:  # TODO sort this oot
        # run the get_reward_noise for each action choice, can probs vmap
        actions = jnp.expand_dims(jnp.arange(0, self._action_spec.num_values, step=1), axis=-1)
        actions = jnp.broadcast_to(actions, (actions.shape[0], obs.shape[0]))

        obs = jnp.broadcast_to(obs, (actions.shape[0], *obs.shape))

        reward_over_actions, _ = jax.vmap(self._get_reward_noise, in_axes=(0, 0))(obs, actions)
        # TODO is the above okay since it is a loop?
        reward_over_actions = jnp.swapaxes(jnp.squeeze(reward_over_actions, axis=-1), 0, 1)

        return reward_over_actions

    def update(self,
            timestep: dm_env.TimeStep,
            action: base.Action,
            logits,
            new_timestep: dm_env.TimeStep,
            buffer_state,
    ):
        """Adds a transition to the trajectory buffer and periodically does SGD."""
        mask = np.random.binomial(1, self._mask_prob, self._num_ensemble)
        noise = np.random.randn(self._num_ensemble)

        self._buffer.append(timestep, action, logits, new_timestep, mask, noise)

        # self._buffer.append(timestep, action, logits, new_timestep, mask, noise)  # TODO ignore this dodgyness for now
        # if self._buffer.full() or new_timestep.last():
        if new_timestep.last():
            trajectory = self._buffer.drain()
            # buffer_data = {"obs": trajectory.observations,
            #                "actions": jnp.expand_dims(trajectory.actions, axis=-1),
            #                "logits": trajectory.logits,
            #                "rewards": jnp.expand_dims(trajectory.rewards, axis=-1),
            #                "discounts": jnp.expand_dims(trajectory.discounts, axis=-1),
            #                "step": jnp.expand_dims(trajectory.step, axis=-1),
            #                "mask": trajectory.mask,
            #                "noise": trajectory.noise,
            #                }
            # broadcast_fn = lambda x: jnp.broadcast_to(x, (2, *x.shape))  # add batch dim
            # fake_batch_sequence = jax.tree_util.tree_map(broadcast_fn, buffer_data)
            # # buffer_state = self._fbx_buffer.add(buffer_state,
            # #                                     fake_batch_sequence
            # #                                     )
            # # batch = self._fbx_buffer.sample(buffer_state, _key)

            state_action_reward_noise, reward_pred = self._get_reward_noise(trajectory.observations[:-1],
                                                                            trajectory.actions)
            state_reward_noise = self._reward_noise_over_actions(trajectory.observations[:-1])

            self._state, pv_loss, buffer_state = self._sgd_step(self._state,
                                                                                      buffer_state,
                                                                                      trajectory,
                                                                state_action_reward_noise,
                                                                state_reward_noise)

            ensemble_loss_all = jnp.zeros((self._num_ensemble,))
            for k, ensemble_state in enumerate(self._ensemble):
                transitions = [trajectory.observations[:-1], trajectory.actions, trajectory.rewards,
                               trajectory.mask[:, k], trajectory.noise[:, k]]
                # TODO is this right observations [:-1]
                self._ensemble[k], ensemble_loss_ind = self._ensemble_sgd_step(ensemble_state, transitions)
                ensemble_loss_all = ensemble_loss_all.at[k].set(ensemble_loss_ind)

            def callback(pv_loss, ensemble_loss_all, reward_pred):
                metric_dict = {"policy_and_value_loss": pv_loss,
                               # "model_params": first_ensemble
                               }
                for ensemble_id, _ in enumerate(self._ensemble):
                    metric_dict[f"Ensemble_{ensemble_id}_Reward_Pred_pv"] = reward_pred[ensemble_id, 6]
                    metric_dict[f"Ensemble_{ensemble_id}_Loss"] = ensemble_loss_all[ensemble_id]

                wandb.log(metric_dict)

                for ensemble_id, _ in enumerate(self._ensemble):
                    wandb.log({f"Ensemble_{ensemble_id}_Loss": ensemble_loss_all[ensemble_id]})


            jax.experimental.io_callback(callback, None, pv_loss,
                                         ensemble_loss_all, reward_pred)
            # TODO I have added wandb stuff in wrappers as well, not really a todo more of a note


        return buffer_state


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
        importance_sampling_exponent=0.995
    )
