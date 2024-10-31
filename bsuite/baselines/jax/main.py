import wandb
import jax
from jax.lib import xla_bridge
from ml_collections import config_dict
from absl import flags, app
import sys
from .vapor_lite.run import run as vlite_run
from .ersac.run import run as ersac_run

# Experiment flags.
_UNCERTAINTY_SCALE = flags.DEFINE_float('uncertainty_scale', 1.0, 'uncertainty_scale')
_MASK_PROB = flags.DEFINE_float('mask_prob', 0.8, 'mask prob')
_HIDDEN_SIZE = flags.DEFINE_integer('hidden_size', 50, 'hidden size')
_PRIOR_SCALE = flags.DEFINE_float('prior_scale', 1.0, 'prior scale')
_LR = flags.DEFINE_float('lr', 1e-3, 'lr')
_ENS_LR = flags.DEFINE_float('ens_lr', 1e-3, 'ens lr')
_TAU_LR = flags.DEFINE_float('tau_lr', 1e-3, 'tau lr')
_DEEP_SEA_SIZE = flags.DEFINE_integer('deep_sea_size', 1, 'Deep sea size')
_NUM_EPS = flags.DEFINE_integer('num_episodes', 25000, 'Overrides number of training eps.')
_SEED = flags.DEFINE_integer('seed', 42, "random seed")
_ALGO = flags.DEFINE_string("algo", "ERSAC", "algorithm used")
_OFF_POLICY = flags.DEFINE_bool("off_policy", False, "off policy or not")
_PPO = flags.DEFINE_bool("ppo", True, "iff ppo or not")
_DISABLE_JIT = flags.DEFINE_bool("disable_jit", False, "to disable jit or not")
# _WANDB_MODE = flags.DEFINE_string("wandb_mode", "disabled", "wandb enabled or not")

def main(_):
    config = config_dict.ConfigDict()

    config.GAMMA = 0.99
    config.TD_LAMBDA = 0.8
    config.REWARD_NOISE_SCALE = 0.1  # set in the ersac paper

    config.UNCERTAINTY_SCALE = _UNCERTAINTY_SCALE.value
    config.MASK_PROB = _MASK_PROB.value
    config.HIDDEN_SIZE = _HIDDEN_SIZE.value
    config.PRIOR_SCALE = _PRIOR_SCALE.value
    config.LR = _LR.value
    config.ENS_LR = _ENS_LR.value
    config.TAU_LR = _TAU_LR.value
    config.DEEP_SEA_MAP = _DEEP_SEA_SIZE.value
    config.NUM_EPISODES = _NUM_EPS.value
    config.SEED = _SEED.value
    config.ALGO = _ALGO.value
    config.OFF_POLICY = _OFF_POLICY.value
    config.PPO = _PPO.value
    config.DISABLE_JIT = _DISABLE_JIT.value

    config.BSUITE_ID = 'deep_sea/1'[0:9] + str(config.DEEP_SEA_MAP)

    config.LOG_EVERY = True  # to log every episode or not

    config.ROLLOUT_LEN = int(10 + (2 * config.DEEP_SEA_MAP) + 5)  # +10 is an extra help should check this doesn't do anything tbh

    config.DEVICE = xla_bridge.get_backend().platform
    print(config.DEVICE)

    with jax.disable_jit(disable=config.DISABLE_JIT):
        if config.ALGO == "VLITE":
            wandb.init(project="BSuite_Testing",
                       # entity=config.WANDB_ENTITY,
                       config=config,
                       group="vlite_testing",
                       # mode=_WANDB_MODE.value
                       )
            vlite_run(config)

        elif config.ALGO == "ERSAC":
            wandb.init(project="BSuite_Testing",
                       # entity=config.WANDB_ENTITY,
                       config=config,
                       group="ersac_testing",
                       # mode=_WANDB_MODE.value
                       )
            ersac_run(config)

        else:
            print("NAH")
            sys.exit(0)

    print("FINITO")


if __name__ == '__main__':
    app.run(main)

