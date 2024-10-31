#!/usr/bin/env python3
# type: ignore
from absl import app
from absl import flags
from absl import logging
from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.contrib import ucl
from ml_collections import config_flags
import sys
import os
import importlib


with open("../multi_agent_climate_pathways/wandb_api_key.txt", "r") as file:
    wandb_api_key = file.read().strip()

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_USE_GPU = flags.DEFINE_boolean("use_gpu", None, "If set, use GPU")
_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "container", None, "Path to singularity container"
)
_EXP_NAME = flags.DEFINE_string("exp_name", "DEEPSEA_TESTS", "Name of experiment")
_ENTRYPOINT = flags.DEFINE_string("entrypoint", None, "Entrypoint for experiment")

_SWEEP = flags.DEFINE_string("sweep", "SWEEP", "Name of the sweep")
# _SWEEP = flags.DEFINE_string("sweep", None, "Name of the sweep")

_SWEEP_INDEX = flags.DEFINE_string("sweep_index", None, "Index of configuration in the sweep")
# _SWEEP_INDEX = flags.DEFINE_string("sweep_index", "0", "Index of configuration in the sweep")

_WANDB_GROUP = flags.DEFINE_string("wandb_group", "{xid}_{name}", "wandb group")
_WANDB_PROJECT = flags.DEFINE_string("wandb_project", "ProbInfMarl",
                                     "wandb project")
_WANDB_ENTITY = flags.DEFINE_string("wandb_entity", "jamesr-j", "wandb entity")
_WANDB_MODE = flags.DEFINE_string("wandb_mode", "online", "wandb mode")

config_flags.DEFINE_config_file("config", None, "Path to config")
flags.mark_flags_as_required(["config", "entrypoint", "use_gpu"])
FLAGS = flags.FLAGS


def _get_hyper():
    if _SWEEP.value is not None:
        sweep_file = config_flags.get_config_filename(FLAGS["config"])
        sys.path.insert(0, os.path.abspath(os.path.dirname(sweep_file)))
        sweep_module, _ = os.path.splitext(os.path.basename(sweep_file))
        m = importlib.import_module(sweep_module)
        sys.path.pop(0)
        sweep_fn_name = f"sweep_{_SWEEP.value}"
        logging.info(f"Running sweep {sweep_fn_name}")
        sweep_fn = getattr(m, sweep_fn_name, None)
        if sweep_fn is None:
            raise ValueError(f"Sweep {sweep_fn_name} does not exist in {sweep_file}")
        else:
            return sweep_fn()
    else:
        return [{}]


def main(_):
    exp_name = _EXP_NAME.value
    if exp_name is None:
        exp_name = _ENTRYPOINT.value.replace(".", "_")
    with xm_cluster.create_experiment(experiment_title=exp_name) as experiment:
        env_vars = {}
        if _USE_GPU.value:
            job_requirements = xm_cluster.JobRequirements(gpu=1, ram=64 * xm.GB)
        else:
            job_requirements = xm_cluster.JobRequirements(ram=32 * xm.GB)
            env_vars['JAX_PLATFORMS'] = 'cpu'

        env_vars["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env_vars["JAX_TRACEBACK_FILTERING"] = "off"
        env_vars["XLA_PYTHON_CLIENT_MEM_FRACTION"] = .95
        env_vars["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # allocates memory when it is needed

        if _LAUNCH_ON_CLUSTER.value:
            # This is a special case for using SGE in UCL where we use generic
            # job requirements and translate to SGE specific requirements.
            # Non-UCL users, use `xm_cluster.GridEngine directly`.
            orbax_dir = "/cluster/project0/orbax"
            executor = ucl.UclGridEngine(
                job_requirements,
                walltime=12 * xm.Hr,  # 48 is max
                # extra_directives=["-l gpu_type=rtx4090"],
                # extra_directives=["-l gpu_type=rtx4090 -pe gpu 3"],  # TODO allows specifying multiple GPUS
                # extra_directives=["-l gpu_type=gtx1080ti"],  # TODO for beaker  https://hpc.cs.ucl.ac.uk/gpus/
                # extra_directives=["-ac allow=EF"],  # TODO for myriad  https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/
                # singularity_options=xm_cluster.SingularityOptions(bind={orbax_dir: orbax_dir}),
            )
            env_vars["ORBAX_DIR"] = orbax_dir
        else:
            FLAGS.wandb_mode = "disabled"
            orbax_dir = "/mnt/cluster/project0/orbax"
            executor = xm_cluster.Local(job_requirements,
                                        singularity_options=xm_cluster.SingularityOptions(bind={orbax_dir: orbax_dir}))
            env_vars["ORBAX_DIR"] = orbax_dir
            # env_vars['JAX_PLATFORMS'] = 'cpu'
        env_vars["LAUNCH_ON_CLUSTER"] = _LAUNCH_ON_CLUSTER.value
        spec = xm_cluster.PythonPackage(
            # This is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            path=".",
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName(_ENTRYPOINT.value),
        )

        # Wrap the python_package to be executing in a singularity container.
        singularity_container = _SINGULARITY_CONTAINER.value

        # It's actually not necessary to use a container, without it, we
        # fallback to the current python environment for local executor and
        # whatever Python environment picked up by the cluster for GridEngine.
        # For remote execution, using the host environment is not recommended.
        # as you may spend quite some time figuring out dependency problems than
        # writing a simple Dockfiler/Singularity file.
        if singularity_container is not None:
            spec = xm_cluster.SingularityContainer(
                spec,
                image_path=singularity_container,
            )

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec(), env_vars=env_vars)]
        )

        config_resource = xm_cluster.Fileset(files={config_flags.get_config_filename(FLAGS["config"]): "lxm3_config.py"})
        args = {"config": config_resource.get_path("lxm3_config.py", executor.Spec())}  # type: ignore
        overrides = config_flags.get_override_values(FLAGS["config"])
        overrides = {f"config.{k}": v for k, v in overrides.items()}
        logging.info("Overrides: %r", overrides)
        args.update(overrides)
        sweep = list(_get_hyper())
        if _SWEEP_INDEX.value is not None:
            filtered_sweep = []
            for index in _SWEEP_INDEX.value.split(","):
                filtered_sweep.append(sweep[int(index)])
            sweep = filtered_sweep
        logging.info("Will launch %d jobs", len(sweep))
        print(sweep)

        """
        SINGLE RUN BELOW
        """
        # xid = experiment.experiment_id
        # experiment_name = exp_name
        # experiment.add(
        #     xm.Job(
        #         executable=executable,
        #         executor=executor,
        #         # You can pass additional arguments to your executable with args
        #         # This will be translated to `--seed 1`
        #         # Note for booleans we currently use the absl.flags convention
        #         # so {'gpu': False} will be translated to `--nogpu`
        #         # args={"seed": 1},
        #         # You can customize environment_variables as well.
        #         # args=sweep[0],
        #         env_vars={"WANDB_API_KEY": wandb_api_key,
        #         "WANDB_PROJECT": _WANDB_PROJECT.value,
        #         "WANDB_ENTITY": _WANDB_ENTITY.value,
        #         "WANDB_NAME": f"{experiment_name}_{xid}_{0}",
        #         "WANDB_MODE": _WANDB_MODE.value,
        #         # "WANDB_RUN_GROUP": _WANDB_GROUP.value.format(name=experiment_name, xid=xid),
        #         "WANDB_RUN_GROUP": experiment_name}
        #     )
        # )

        """ 
        BATCH RUN BELOW
        """
        xid = experiment.experiment_id
        experiment_name = exp_name
        envs = [
            {
                "WANDB_API_KEY": wandb_api_key,
                "WANDB_PROJECT": _WANDB_PROJECT.value,
                "WANDB_ENTITY": _WANDB_ENTITY.value,
                "WANDB_NAME": f"{experiment_name}_{xid}_{wid+1}",
                "WANDB_MODE": _WANDB_MODE.value,
                # "WANDB_RUN_GROUP": _WANDB_GROUP.value.format(name=experiment_name, xid=xid),
                "WANDB_RUN_GROUP": experiment_name,
            } for wid in range(len(sweep))
        ]
        experiment.add(
            xm_cluster.ArrayJob(executable, executor, args=sweep, env_vars=envs)
        )


if __name__ == "__main__":
    app.run(main)
