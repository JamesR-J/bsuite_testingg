import sys
import itertools
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.LR = 2.5e-4
    config.NUM_ENVS = 128
    config.NUM_STEPS = 256
    config.TOTAL_TIMESTEPS = 100
    config.UPDATE_EPOCHS = 4
    config.NUM_MINIBATCHES = 4
    config.GAMMA = 0.99
    config.GAE_LAMBDA = 0.95
    config.CLIP_EPS = 0.2
    config.ENT_COEF = 0.5
    config.VF_COEF = 0.5
    config.MAX_GRAD_NORM = 0.5
    config.ACTIVATION = "tanh"
    config.ANNEAL_LR = True
    config.GRU_HIDDEN_DIM = 256
    config.SCALE_CLIP_EPS = False

    config.NUM_AGENTS = 2
    config.REWARD_TYPE = ["PB"]
    config.AGENT_TYPE = ["PPO"]

    config.HOMOGENEOUS = False

    config.RUN_TRAIN = True
    config.RUN_EVAL = False
    config.NUM_EVAL_STEPS = 2000

    return config  # TODO get this to work at some point

def sweep_SWEEP():
    uncertainty_scale_list = [0.01, 0.1, 1.0, 2.0, 10.0, 100.0]
    mask_prob_list = [0.6, 0.8, 1.0]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hidden_size_list = [50]  # [50, 64, 128, 256, 512]
    prior_scale_list = [3.0]  # [0.1, 1.0, 2.0, 3.0, 4.0]
    lr_list = [1e-3]  # [1e-2, 1e-3, 1e-4, 1e-5]
    ens_lr_list = [1e-3]  # [1e-2, 1e-3, 1e-4, 1e-5]
    tau_lr_list = [1e-3]  # [1e-2, 1e-3, 1e-4, 1e-5]
    deep_sea_size_list = [2, 4, 6, 8, 10]
    num_eps_list = [250]  # 25000
    seed_list = [28, 10, 98]  # , 44, 22, 68]

    algo_list = ["ERSAC"]

    combinations = itertools.product(uncertainty_scale_list, mask_prob_list, hidden_size_list, prior_scale_list, lr_list, ens_lr_list, tau_lr_list,
                                     deep_sea_size_list, num_eps_list, seed_list, algo_list)
    result = [{"uncertainty_scale": uncertainty_scale,
               "mask_prob": mask_prob,
               "hidden_size": hidden_size,
               "prior_scale": prior_scale,
               "lr": lr,
               "ens_lr": ens_lr,
               "tau_lr": tau_lr,
               "deep_sea_size": deep_sea_size,
               "num_episodes": num_eps,
               "seed": seed,
               "algo": algo,
               "disable_jit": False} for uncertainty_scale, mask_prob, hidden_size, prior_scale, lr, ens_lr, tau_lr,
              deep_sea_size, num_eps, seed, algo in combinations]

    return result

# def get_sweep():
#     """Returns a sweep configuration for hyperparameter tuning."""
#
#     config = config_dict.ConfigDict()
#     config.learning_rate = 0.001
#     sweep_config.params["batch_size"] = config_dict.randint(16, 64)
#     # Add other hyperparameters with sweep ranges
#     return sweep
