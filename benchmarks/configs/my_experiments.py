import os
from dataclasses import asdict

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MotorSystemConfig,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.simulators.habitat.configs.voltage_touch_dataset import (
    VoltageTouchDatasetArgs,
    get_voltage_touch_dataloader_args,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.fontys_finger_pressure_environment import VoltageTouchEnvironment
from tbp.monty.frameworks.experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.graph_matching import GraphLM
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    PressureTouchSensorSM,
)

"""
Basic setup
-----------
"""
# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

"""
Training
----------------------------------------------------------------------------------------
"""
voltage_touch_test = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=1,
        do_eval=False,
        max_train_steps=100,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=project_dir,
        run_name="voltage_touch_test",
        wandb_handlers=[],
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=50),
        sensor_module_configs=dict(
            sensor_module_0=dict(
                sensor_module_class=PressureTouchSensorSM,
                sensor_module_args=dict(
                    sensor_module_id="finger",
                    features=["voltage", "voltage_history"],
                    save_raw_obs=True,
                ),
            ),
        ),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=GraphLM,
                learning_module_args=dict(),
            )
        ),
        motor_system_config=MotorSystemConfig(),
        sm_to_agent_dict=dict(finger="agent_id_0"),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=VoltageTouchDatasetArgs(
        env_init_func=VoltageTouchEnvironment,
        env_init_args={}, # empty because environment has no arguments
    ),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=get_voltage_touch_dataloader_args(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=get_voltage_touch_dataloader_args(),
)

experiments = MyExperiments(
    voltage_touch_test=voltage_touch_test,
)
CONFIGS = asdict(experiments)