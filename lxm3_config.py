import sys
import itertools
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

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
    num_eps_list = [25000]
    seed_list = [28, 10, 98]  # , 44, 22, 68]

    algo_list = ["ERSAC"]
    off_policy_list = [False]
    ppo_list = [False]

    combinations = itertools.product(uncertainty_scale_list, mask_prob_list, hidden_size_list, prior_scale_list,
                                     lr_list, ens_lr_list, tau_lr_list, deep_sea_size_list, num_eps_list, seed_list,
                                     algo_list, off_policy_list, ppo_list)
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
               "off_policy": off_policy,
               "ppo": ppo,
               "disable_jit": False} for uncertainty_scale, mask_prob, hidden_size, prior_scale, lr, ens_lr, tau_lr,
              deep_sea_size, num_eps, seed, algo, off_policy, ppo in combinations]

    return result
