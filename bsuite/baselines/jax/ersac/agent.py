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
import wandb

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
    opt_state: Any
    step: int


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: Any
    # tau: float
    tau_params: hk.Params
    tau_opt_state: Any


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
            tau_optimizer: optax.GradientTransformation,
            rng: hk.PRNGSequence,
            sequence_length: int,
            discount: float,
            td_lambda_val: float,
            reward_noise_scale: float,
            init_tau: float,
            num_ensemble: int
    ):
        def _get_reward_noise(obs, actions):
            ensemble_obs = jnp.repeat(obs[jnp.newaxis, :], num_ensemble, axis=0)
            ensemble_action = jnp.repeat(actions[jnp.newaxis, :], num_ensemble, axis=0)
            # # TODO check ye old ensemble obs and action and why not using them

            def single_reward_noise(state, obs, action):
                reward_pred =  ensemble_network.apply(state.params, obs, jnp.expand_dims(action, axis=-1))
                return reward_pred

            # ensembled_reward = jnp.zeros((num_ensemble, actions.shape[0], 1))
            # for k, state in enumerate(self._ensemble):
            #     ensembled_reward = ensembled_reward.at[k].set(single_reward_noise(state, obs, actions))
            ensembles = jax.tree_util.tree_map(lambda x: x, self._ensemble)
            ensembled_reward = jax.vmap(single_reward_noise)()

            SIGMA_SCALE = 3.0
            ensembled_reward = SIGMA_SCALE * jnp.std(ensembled_reward, axis=0)
            # ensembled_reward = SIGMA_SCALE * jnp.square(jnp.std(ensembled_reward, axis=0))
            ensembled_reward = jnp.minimum(ensembled_reward, 1)
            # ensembled_reward = jnp.square(jnp.std(ensembled_reward, axis=0))

            return ensembled_reward

        # Define loss function.
        def loss(trajectory: sequence.Trajectory, tau_params) -> jnp.ndarray:
            tau = jnp.exp(tau_params)

            """"Actor-critic loss."""
            logits, values = network(trajectory.observations)
            policy_dist = distrax.Softmax(logits=logits[:-1])
            log_prob = policy_dist.log_prob(trajectory.actions)

            state_action_reward_noise = _get_reward_noise(trajectory.observations[:-1], trajectory.actions)

            td_lambda = jax.vmap(rlax.td_lambda, in_axes=(1, 1, 1, 1, None), out_axes=1)
            k_estimate = td_lambda(jnp.expand_dims(values[:-1], axis=-1),
                                   jnp.expand_dims(trajectory.rewards, axis=-1) + (
                                               jnp.square(state_action_reward_noise) / 2 * tau),
                                   jnp.expand_dims(trajectory.discounts * discount, axis=-1),
                                   jnp.expand_dims(values[1:], axis=-1),
                                   jnp.array(td_lambda_val),
                                   )

            value_loss = jnp.mean(jnp.square(values[:-1] - jax.lax.stop_gradient(k_estimate - tau * log_prob)))
            # TODO is it right to use [1:] for these values etc or [:-1]?

            entropy = policy_dist.entropy()

            policy_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(k_estimate - values[:-1] - tau * entropy))

            # state_action_reward_noise = _get_reward_noise(trajectory.observations[:-1], trajectory.actions)
            # td_errors = rlax.td_lambda(
            #     v_tm1=values[:-1],
            #     r_t=trajectory.rewards + (jnp.square(jnp.squeeze(state_action_reward_noise, axis=-1)) / (2 * tau)),
            #     discount_t=trajectory.discounts * discount,
            #     v_t=values[1:],
            #     lambda_=jnp.array(td_lambda_val),
            # )
            # value_loss = jnp.mean(td_errors ** 2)
            # policy_loss = rlax.policy_gradient_loss(
            #     logits_t=logits[:-1],
            #     a_t=trajectory.actions,
            #     adv_t=td_errors,
            #     w_t=jnp.ones_like(td_errors))
            # # policy_loss = jnp.zeros(())
            # # value_loss = jnp.zeros(())
            # entropy = jnp.zeros(())

            return policy_loss + value_loss, entropy

        def tau_loss(log_tau, trajectory: sequence.Trajectory, entropy) -> jnp.ndarray:
            tau = jnp.exp(log_tau)
            state_action_reward_noise = _get_reward_noise(trajectory.observations[:-1], trajectory.actions)

            tau_loss = (jnp.square(state_action_reward_noise) / (2 * tau)) + tau * entropy

            return jnp.mean(tau_loss)

        # Transform the loss into a pure function.
        loss_fn = hk.without_apply_rng(hk.transform(loss)).apply

        # Transform the (impure) network into a pure function.
        ensemble_network = hk.without_apply_rng(hk.transform(ensemble_network))

        # Define update function.
        @jax.jit
        def sgd_step(state: TrainingState,
                     trajectory: sequence.Trajectory,
                     bsuite_info) -> TrainingState:
            """Does a step of SGD over a trajectory."""
            (pv_loss, entropy), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params, trajectory,
                                                                                      state.tau_params)
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

            tau_loss_val, tau_gradients = jax.value_and_grad(tau_loss)(state.tau_params, trajectory, entropy)
            tau_updates, new_tau_opt_state = tau_optimizer.update(tau_gradients, state.tau_opt_state)
            new_tau_params = optax.apply_updates(state.tau_params, tau_updates)
            tau = jnp.exp(new_tau_params)

            ensemble_loss_all = jnp.zeros((self._num_ensemble,))
            for k, ensemble_state in enumerate(self._ensemble):
                transitions = [trajectory.observations[:-1], trajectory.actions, trajectory.rewards,
                               trajectory.mask[:, k], trajectory.noise[:, k]]
                # TODO is this right observations [:-1]
                self._ensemble[k] = self._ensemble_sgd_step(ensemble_state, transitions)
                # self._ensemble[k], ensemble_loss_ind = self._ensemble_sgd_step(ensemble_state, transitions)
                # ensemble_loss_all = ensemble_loss_all.at[k].set(ensemble_loss_ind)

            def callback(pv_loss, tau, tau_loss_val, ensemble_loss_all, info):
                metric_dict = {"policy_and_value_loss": pv_loss,
                               "tau": tau,
                               "tau_loss": tau_loss_val,
                               # "ensemble_loss": ensemble_loss_all,
                               "denoised_return": info["denoised_return"],
                               "episode_return": info["episode_return"],
                               "episode": info["episode"]
                               }

                wandb.log(metric_dict)

                for ensemble_id in range(self._num_ensemble):
                    wandb.log({f"Ensemble_{ensemble_id}_Loss": ensemble_loss_all[k]})

            jax.experimental.io_callback(callback, None, pv_loss, tau, tau_loss_val, ensemble_loss_all, bsuite_info)

            # TODO add wandb stuff here
            # critic+actor loss
            # ensemble loss
            # tau
            # tau loss
            # episode return
            # denoised return

            return TrainingState(params=new_params, opt_state=new_opt_state, tau_params=new_tau_params,
                                 tau_opt_state=new_tau_opt_state)

        # Define update function for each member of ensemble.
        @jax.jit
        def ensemble_sgd_step(state: EnsembleTrainingState,
                              transitions: Sequence[jnp.ndarray]) -> Tuple[EnsembleTrainingState, Any]:
            """Does a step of SGD for the whole ensemble over `transitions`."""

            ensemble_loss_val, gradients = jax.value_and_grad(ensemble_loss)(state.params, transitions)
            updates, new_opt_state = ensemble_optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

            return EnsembleTrainingState(params=new_params,
                                         opt_state=new_opt_state,
                                         step=state.step + 1)
                    # ensemble_loss_val)

        # Define loss function, including bootstrap mask `m_t` & reward noise `z_t`.
        def ensemble_loss(params: hk.Params,
                          transitions: Sequence[jnp.ndarray]) -> jnp.ndarray:
            """Q-learning loss with added reward noise + half-in bootstrap."""
            o_tm1, a_tm1, r_t, m_t, z_t = transitions
            r_t_pred = ensemble_network.apply(params, o_tm1, jnp.expand_dims(a_tm1, axis=-1))
            # r_t += reward_noise_scale * z_t  # TODO don't need this when on policy
            return 0.5 * jnp.mean(m_t * jnp.square(r_t - jnp.squeeze(r_t_pred, axis=-1)))  # TODO is this right?

        # Initialize network parameters and optimiser state.
        init, forward = hk.without_apply_rng(hk.transform(network))
        dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)
        dummy_action = jnp.zeros((1, 1), dtype=jnp.int32)
        initial_params = init(next(rng), dummy_observation)
        initial_opt_state = optimizer.init(initial_params)

        initial_ensemble_params = [
            ensemble_network.init(next(rng), dummy_observation, dummy_action) for _ in range(num_ensemble)
        ]
        # initial_ensemble_target_params = [
        #     ensemble_network.init(next(rng), dummy_observation, dummy_action) for _ in range(num_ensemble)
        # ]
        initial_ensemble_opt_state = [ensemble_optimizer.init(p) for p in initial_ensemble_params]

        log_tau = jnp.asarray(0., dtype=jnp.float32)
        tau_opt_state = tau_optimizer.init(log_tau)

        # Internalize state.
        self._state = TrainingState(initial_params, initial_opt_state, log_tau, tau_opt_state)
        self._ensemble = [
            EnsembleTrainingState(p, o, step=0) for p, o in zip(
                initial_ensemble_params,
                # initial_ensemble_target_params,
                initial_ensemble_opt_state)
        ]
        self._forward = jax.jit(forward)
        self._ensemble_forward = jax.jit(ensemble_network.apply)
        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length, num_ensemble)
        self._sgd_step = sgd_step
        self._ensemble_sgd_step = ensemble_sgd_step
        self._rng = rng
        self._num_ensemble = num_ensemble
        self._mask_prob = 1.0
        self._obs_spec = obs_spec
        self._init_tau = init_tau

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
               buffer_state,
               bsuite_info
               ):
        """Adds a transition to the trajectory buffer and periodically does SGD."""
        mask = np.random.binomial(1, self._mask_prob, self._num_ensemble)
        noise = np.random.randn(self._num_ensemble)

        self._buffer.append(timestep, action, logits, new_timestep, mask, noise)

        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()
            self._state = self._sgd_step(self._state, trajectory, bsuite_info)

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
        # x = hk.Linear(50)(obs)  # TODO curr jut obs and not actions together
        # actions = jax.lax.convert_element_type(actions, new_dtype=jnp.float32)
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
        tau_optimizer=optax.adam(3e-4),
        rng=hk.PRNGSequence(seed),
        sequence_length=50,
        discount=0.99,
        td_lambda_val=0.9,
        reward_noise_scale=1.0,
        num_ensemble=10,
        init_tau=0.001  # 0.0001
    )
