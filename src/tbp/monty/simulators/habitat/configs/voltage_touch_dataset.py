from dataclasses import dataclass
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.environments.fontys_finger_pressure_environment import VoltageTouchEnvironment

@dataclass
class VoltageTouchDatasetArgs:
    """Configuration for voltage touch sensor dataset."""
    env_init_func: callable = VoltageTouchEnvironment
    env_init_args: dict = None
    
    def __post_init__(self):
        if self.env_init_args is None:
            self.env_init_args = {}