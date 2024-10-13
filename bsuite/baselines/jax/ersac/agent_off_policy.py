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
from functools import partial
import flashbax

Array = chex.Array
Logits = jnp.ndarray
Value = jnp.ndarray
PolicyValueNet = Callable[[jnp.ndarray], Tuple[Logits, Value]]


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
            mask_prob: float,
            init_tau: float,
            num_ensemble: int,
            batch_size: int
    ):
        # Define loss function.
        def loss(batch: Any, tau_params, state_action_reward_noise) -> jnp.ndarray:
            tau = jnp.exp(tau_params)

            """"Actor-critic loss."""
            logits, values = hk.BatchApply(network)(batch.experience["obs"])

            def get_log_prob(logits, actions):
                dist = distrax.Softmax(logits)
                return dist.log_prob(actions), dist.entropy()

            log_prob, entropy = jax.vmap(get_log_prob)(logits[:, :-1], batch.experience["actions"][:, :-1])

            td_lambda = jax.vmap(rlax.td_lambda, in_axes=(1, 1, 1, 1, None), out_axes=1)
            k_estimate = td_lambda(values[:, :-1],
                                   jnp.squeeze(batch.experience["rewards"][:, :-1] + (
                                           state_action_reward_noise / (2 * tau)), axis=-1),
                                   jnp.squeeze(batch.experience["discounts"][:, :-1] * discount, axis=-1),
                                   values[:, 1:],
                                   td_lambda_val,
                                   )

            # rhos = rlax.categorical_importance_sampling_ratios(logits[:, :-1],
            #                                                    batch.experience["logits"][:, :-1],
            #                                                    batch.experience["actions"][:, :-1])
            # vtrace_td_error = jax.vmap(rlax.vtrace, in_axes=(1, 1, 1, 1, 1, None), out_axes=1)
            # k_estimate = vtrace_td_error(values[:, :-1],
            #                                                values[:, 1:],
            #                                                jnp.squeeze(batch.experience["rewards"][:, :-1] + (
            #                                                            state_action_reward_noise / (2 * tau)), axis=-1),
            #                                                jnp.squeeze(batch.experience["discounts"][:, :-1] * discount, axis=-1),
            #                                                rhos,
            #                                                0.9)

            value_loss = jnp.mean(jnp.square(values[:, :-1] - jax.lax.stop_gradient(k_estimate - tau * log_prob)), axis=-1)
            # TODO is it right to use [1:] for these values etc or [:-1]?

            # entropy = policy_dist.entropy()

            policy_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(k_estimate - values[:, :-1] - tau * entropy), axis=-1)
            # TODO unsure if above correct, true to pseudo code but I think the log_prob should also multiply the entropy potentially

            # TODO add re prioritisation, cba to do at this moment in time
            return jnp.mean(policy_loss) + jnp.mean(value_loss), entropy

        def tau_loss(log_tau, trajectory: sequence.Trajectory, entropy, state_action_reward_noise) -> jnp.ndarray:
            tau = jnp.exp(log_tau)

            tau_loss = jnp.mean((jnp.squeeze(state_action_reward_noise) / (2 * tau)) + (tau * entropy), axis=-1)

            return jnp.mean(tau_loss)

        # Define loss function, including bootstrap mask `m_t` & reward noise `z_t`.
        def ensemble_loss(params: hk.Params,
                          transitions: Sequence[jnp.ndarray]) -> jnp.ndarray:
            """Q-learning loss with added reward noise + half-in bootstrap."""
            o_tm1, a_tm1, r_t, m_t, z_t = transitions
            r_t_pred = ensemble_network.apply(params, o_tm1, jnp.expand_dims(a_tm1, axis=-1))
            r_t += reward_noise_scale * jnp.expand_dims(z_t, axis=-1)
            loss = 0.5 * jnp.mean(m_t * jnp.square(jnp.squeeze(r_t, axis=-1) - jnp.squeeze(r_t_pred, axis=-1)), axis=-1)

            return jnp.mean(loss)

        # Transform the loss into a pure function.
        loss_fn = hk.without_apply_rng(hk.transform(loss)).apply

        # Transform the (impure) network into a pure function.
        ensemble_network = hk.without_apply_rng(hk.transform(hk.BatchApply(ensemble_network)))

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
                                         step=state.step + 1), ensemble_loss_val

        # Define update function.
        @jax.jit
        def sgd_step(state: TrainingState,
                     trajectory: sequence.Trajectory,
                     state_action_reward_noise) -> TrainingState:
            """Does a step of SGD over a trajectory."""
            (pv_loss, entropy), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params, trajectory,
                                                                                      state.tau_params,
                                                                                      state_action_reward_noise)
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

            tau_loss_val, tau_gradients = jax.value_and_grad(tau_loss, has_aux=False)(state.tau_params, trajectory,
                                                                                      entropy,
                                                                                      state_action_reward_noise)
            tau_updates, new_tau_opt_state = tau_optimizer.update(tau_gradients, state.tau_opt_state)
            new_tau_params = optax.apply_updates(state.tau_params, tau_updates)
            tau = jnp.exp(new_tau_params)

            return (TrainingState(params=new_params, opt_state=new_opt_state, tau_params=new_tau_params,
                                 tau_opt_state=new_tau_opt_state), pv_loss, tau, tau_loss_val)


        # Initialize network parameters and optimiser state.
        init, forward = hk.without_apply_rng(hk.transform(hk.BatchApply(network)))
        dummy_observation = jnp.zeros((batch_size, 1, *obs_spec.shape), dtype=jnp.float32)
        dummy_action = jnp.zeros((batch_size, 1, 1), dtype=jnp.int32)
        initial_params = init(next(rng), dummy_observation)
        initial_opt_state = optimizer.init(initial_params)

        # dummy_ens_observation = jnp.broadcast_to(dummy_observation, (batch_size, *dummy_observation.shape))
        # dummy_ens_action = jnp.broadcast_to(dummy_action, (batch_size, *dummy_action.shape))

        initial_ensemble_params = [
            ensemble_network.init(next(rng), dummy_observation, dummy_action) for _ in range(num_ensemble)
        ]
        initial_ensemble_opt_state = [ensemble_optimizer.init(p) for p in initial_ensemble_params]

        log_tau = jnp.asarray(jnp.log(init_tau), dtype=jnp.float32)
        # log_tau = jnp.asarray(init_tau, dtype=jnp.float32)  # TODO unsure how to init this val
        tau_opt_state = tau_optimizer.init(log_tau)

        sample_seq_length = obs_spec.shape[0]  # TODO needs to be size of env
        self._batch_size = batch_size
        self._fbx_buffer = flashbax.make_prioritised_trajectory_buffer(add_batch_size=1,
                                                                       sample_batch_size=self._batch_size,
                                                                       sample_sequence_length=sample_seq_length+1,
                                                                       period=sample_seq_length+1, # So no overlap in trajs?
                                                                       min_length_time_axis=1,
                                                                       max_size=10000,
                                                                       priority_exponent=1.0
                                                                       )

        # Internalize state.
        self._state = TrainingState(initial_params, initial_opt_state, log_tau, tau_opt_state)
        self._ensemble = [
            EnsembleTrainingState(p, o, step=0) for p, o in zip(
                initial_ensemble_params,
                initial_ensemble_opt_state)
        ]
        self._forward = jax.jit(forward)
        self._ensemble_forward = jax.jit(ensemble_network.apply)
        self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length, num_ensemble)
        self._sgd_step = sgd_step
        self._ensemble_sgd_step = ensemble_sgd_step
        self._rng = rng
        self._num_ensemble = num_ensemble
        self._mask_prob = mask_prob
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self._init_tau = init_tau

    def return_buffer(self):
        fake_timestep = {"obs": jnp.zeros((*self._obs_spec.shape,)),
                         "actions": jnp.zeros((), dtype=self._action_spec.dtype),
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
        batch_obs = jnp.broadcast_to(observation, (self._batch_size, *observation.shape))
        logits, _ = self._forward(self._state.params, batch_obs)
        action = jax.random.categorical(key, logits[0]).squeeze()  # Just take any of the output dims?
        return int(action), logits[0]  # Just take any of the output dims?

    @partial(jax.jit, static_argnums=(0,))
    def _single_reward_noise(self, state, obs, action):
        reward_pred = self._ensemble_forward(state.params, obs, jnp.expand_dims(action, axis=-1))
        return reward_pred

    def _get_reward_noise(self, obs, actions):
        # batch size, num_steps, 1
        ensembled_reward_sep = jnp.zeros((self._num_ensemble, actions.shape[0], actions.shape[1], 1))
        for k, state in enumerate(self._ensemble):
            ensembled_reward_sep = ensembled_reward_sep.at[k].set(self._single_reward_noise(state, obs, actions))

        ensembled_reward = jnp.var(ensembled_reward_sep, axis=0)

        return ensembled_reward, ensembled_reward_sep

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

        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()

            buffer_data = {"obs": trajectory.observations,
                           "actions": jnp.concatenate((trajectory.actions, jnp.zeros((1,), dtype=self._action_spec.dtype))),
                           "logits": jnp.concatenate((trajectory.logits, jnp.zeros((1, 2)))),
                           "rewards": jnp.concatenate((jnp.expand_dims(trajectory.rewards, axis=-1), jnp.zeros((1, 1)))),
                           "discounts": jnp.concatenate((jnp.expand_dims(trajectory.discounts, axis=-1), jnp.zeros((1, 1)))),
                           "step": jnp.concatenate((jnp.expand_dims(trajectory.step, axis=-1), jnp.zeros((1, 1)))),
                           "mask": jnp.concatenate((trajectory.mask, jnp.zeros((1, self._num_ensemble)))),
                           "noise": jnp.concatenate((trajectory.noise, jnp.zeros((1, self._num_ensemble)))),
                           }
            broadcast_fn = lambda x: jnp.broadcast_to(x, (1, *x.shape))  # add batch dim think first dim
            # TODO for testing made the above copy 16 times which is dodgy for now
            fake_batch_sequence = jax.tree_util.tree_map(broadcast_fn, buffer_data)
            buffer_state = self._fbx_buffer.add(buffer_state,
                                                fake_batch_sequence
                                                )
            batch = self._fbx_buffer.sample(buffer_state, next(self._rng))
            # TODO check it gets full trajectories and not random ones, can see this by the zero additions I have put at the end

            state_action_reward_noise, reward_pred = self._get_reward_noise(batch.experience["obs"][:, :-1],
                                                                                batch.experience["actions"][:, :-1])

            self._state, pv_loss, tau, tau_loss_val = self._sgd_step(self._state, batch, state_action_reward_noise)

            ensemble_loss_all = jnp.zeros((self._num_ensemble,))
            for k, ensemble_state in enumerate(self._ensemble):
                # transitions = [trajectory.observations[:, -1], trajectory.actions, trajectory.rewards,
                #                trajectory.mask[:, k], trajectory.noise[:, k]]
                transitions = [batch.experience["obs"][:, :-1], batch.experience["actions"][:, :-1],
                               batch.experience["rewards"][:, :-1],
                               batch.experience["mask"][:, :-1, k], batch.experience["noise"][:, :-1, k]]
                # TODO is this right observations [:-1]
                self._ensemble[k], ensemble_loss_ind = self._ensemble_sgd_step(ensemble_state, transitions)
                ensemble_loss_all = ensemble_loss_all.at[k].set(ensemble_loss_ind)

            def callback(pv_loss, tau, tau_loss_val, ensemble_loss_all, reward_pred, reward_pred_2):
                metric_dict = {"policy_and_value_loss": pv_loss,
                               "tau": tau,
                               "tau_loss": tau_loss_val,
                               # "model_params": first_ensemble
                               }
                for ensemble_id, _ in enumerate(self._ensemble):
                    metric_dict[f"Ensemble_{ensemble_id}_Reward_Pred_pv"] = reward_pred[ensemble_id, 6]
                    metric_dict[f"Ensemble_{ensemble_id}_Reward_Pred_tau"] = reward_pred_2[ensemble_id, 6]

                wandb.log(metric_dict)

                for ensemble_id, _ in enumerate(self._ensemble):
                    wandb.log({f"Ensemble_{ensemble_id}_Loss": ensemble_loss_all[ensemble_id]})

            jax.experimental.io_callback(callback, None, pv_loss, tau, tau_loss_val,
                                         ensemble_loss_all, reward_pred[:, 0, :, :], reward_pred[:, 0, :, :])
            # 0 just to randomly index one of the batches
            # TODO I have added wandb stuff in wrappers as well, not really a todo more of a note

        return buffer_state


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  config,
                  seed: int = 0) -> base.Agent:
    """Creates an actor-critic agent with default hyperparameters."""

    hidden_sizes = [config.HIDDEN_SIZE, config.HIDDEN_SIZE]

    def network(inputs: jnp.ndarray) -> Tuple[Logits, Value]:
        flat_inputs = hk.Flatten()(inputs)
        # torso = hk.nets.MLP([64, 64])
        torso = hk.nets.MLP(hidden_sizes)  # TODO have changed this
        policy_head = hk.Linear(action_spec.num_values)
        # value_head = hk.Linear(action_spec.num_values)
        value_head = hk.Linear(1)
        embedding = torso(flat_inputs)
        logits = policy_head(embedding)
        value = value_head(embedding)
        # return logits, value  #  jnp.squeeze(value, axis=-1)
        return logits, jnp.squeeze(value, axis=-1)

    prior_scale = config.PRIOR_SCALE  # 0.1  # 5.
    # hidden_sizes = [50, 50]

    def ensemble_network(obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Simple Q-network with randomized prior function."""
        net = hk.nets.MLP([*hidden_sizes, 1])
        prior_net = hk.nets.MLP([*hidden_sizes, 1])
        obs = hk.Flatten()(obs)
        obs = hk.Linear(49)(obs)
        # x = hk.Linear(50)(obs)  # TODO curr jut obs and not actions together
        # actions = jax.lax.convert_element_type(actions, new_dtype=jnp.float32)
        # actions = hk.Linear(25)(actions)
        x = jnp.concatenate((obs, actions), axis=-1)  # TODO shall we convert to float and then apply linear layer?
        return net(x) + prior_scale * jax.lax.stop_gradient(prior_net(x))

    return ActorCritic(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
        ensemble_network=ensemble_network,
        optimizer=optax.adam(config.LR),
        ensemble_optimizer=optax.adam(config.ENS_LR),
        tau_optimizer=optax.adam(config.TAU_LR),
        rng=hk.PRNGSequence(seed),
        sequence_length=config.ROLLOUT_LEN,
        discount=config.GAMMA,
        td_lambda_val=config.TD_LAMBDA,
        reward_noise_scale=config.REWARD_NOISE_SCALE,
        mask_prob=config.MASK_PROB,
        num_ensemble=10,
        init_tau=0.001,
        batch_size=16
    )
