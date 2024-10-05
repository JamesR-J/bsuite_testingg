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
"""Run an actor-critic agent instance on a bsuite experiment."""
import jax
from absl import app
from absl import flags

import bsuite
from bsuite import sweep

from bsuite.baselines import experiment
from bsuite.baselines.jax import ersac
from bsuite.baselines.utils import pool

import wandb
from ml_collections import config_dict

# Internal imports.

# Experiment flags.
flags.DEFINE_string(
    'bsuite_id', 'deep_sea/5', 'BSuite identifier. '
                               # 'bsuite_id', 'deep_sea/10', 'BSuite identifier. '
    'This global flag can be used to control which environment is loaded.')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite', 'terminal'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', True, 'overwrite csv logging if found')
flags.DEFINE_integer('num_episodes', 25000, 'Overrides number of training eps.')
# TODO reset this to 25000
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

FLAGS = flags.FLAGS


def run(og_bsuite_id: str) -> str:
  """Runs an A2C agent on a given bsuite environment, logging to CSV."""

  config = config_dict.ConfigDict()
  config.PRIOR_SCALE = 10.0  # 5.0  # 0.5
  config.LR = 1e-3
  config.ENS_LR = 1e-3
  config.TAU_LR = 1e-3
  config.GAMMA = 0.99
  config.TD_LAMBDA = 0.8
  config.REWARD_NOISE_SCALE = 1.0
  config.MASK_PROB = 0.8
  config.DEEP_SEA_MAP = 20

  bsuite_id = og_bsuite_id[0:9] + str(config.DEEP_SEA_MAP)

  wandb.init(project="BSuite_Testing",
             # entity=config.WANDB_ENTITY,
             config=config,
             group="ersac_testing",
             # mode="disabled",
             mode="online",
             )

  env = bsuite.load_and_record(
      bsuite_id=bsuite_id,
      save_path=FLAGS.save_path,
      logging_mode=FLAGS.logging_mode,
      overwrite=FLAGS.overwrite,
  )

  agent = ersac.default_agent(env.observation_spec(), env.action_spec(), config)

  num_episodes = FLAGS.num_episodes or getattr(env, 'bsuite_num_episodes')
  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=num_episodes,
      verbose=FLAGS.verbose)

  return bsuite_id


def main(_):
  # Parses whether to run a single bsuite_id, or multiprocess sweep.
  bsuite_id = FLAGS.bsuite_id

  if bsuite_id in sweep.SWEEP:
    print(f'Running single experiment: bsuite_id={bsuite_id}.')
    run(bsuite_id)

  elif hasattr(sweep, bsuite_id):
    bsuite_sweep = getattr(sweep, bsuite_id)
    print(f'Running sweep over bsuite_id in sweep.{bsuite_sweep}')
    FLAGS.verbose = False
    pool.map_mpi(run, bsuite_sweep)

  else:
    raise ValueError(f'Invalid flag: bsuite_id={bsuite_id}.')


if __name__ == '__main__':
    # with jax.checking_leaks():
    with jax.disable_jit(disable=False):
        app.run(main)
