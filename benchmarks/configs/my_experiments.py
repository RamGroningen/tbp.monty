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
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import glob

default_raw_touch_data_dir = "/Users/rkaundinya/Development/data/measurments_18okt/meetdata_18okt"

''' Helper Functions'''    
# If num_classes <= 0, then load all data
def create_labels_and_get_raw_data_file_names(dir=default_raw_touch_data_dir, num_classes=0):
    # subset select num of classes if specified
    if num_classes > 0:
        sample_folders = os.listdir(dir)[:num_classes]
    else:
        sample_folders = os.listdir(dir)
        
    joined_list = []
    labels_dict = {} # Index label to fabric name e.g. 0 -> A4A
    labels = []
    
    # Make a list of csv files with pressure data
    for idx, sample_folder in enumerate(sample_folders):
        files = os.path.join(dir, sample_folder, "*.csv")
        labels_dict[idx] = sample_folder
        
        # Add the fabric label for every raw pressure reading file
        for i in range(len(glob.glob(files))):
            labels.append(idx)
        
        for file in glob.glob(files):
            joined_list.append(file)
            
    return labels, labels_dict, joined_list

def generate_unique_random_arrays(ranges, sizes, debug=False):
    # Output same order of data every time if debug is true
    if debug:
        np.random.seed(0)
    
    return [np.random.choice(range(0, range_val), size=size, replace=False) 
        for range_val, size in zip(ranges, sizes)]

def add_list_np_arrays_to_offset(nested_list, offset):
    if len(nested_list) != len(offset):
        raise ValueError("Length of nested_list must match length of offset")

    result = []

    for subarray, number in zip(nested_list, offset):
        subarray = np.array(subarray)
        result.append(subarray + number)

    return result

class RawTouchSensorDataset(Dataset):
    def __init__(self, directory, model_params, label_indices=None):
        self.dir = directory
        self.model_params = model_params
        self.labels, self.labels_dict, self.data_file_names = create_labels_and_get_raw_data_file_names\
                                                                (num_classes=model_params.num_classes)
        
        
        if (type(label_indices) == np.ndarray or type(label_indices) == list):
            self.data_label_indices = label_indices
        else:
            self._choose_training_dataset(self.model_params.num_classes)


        if model_params.bin_time <= 0:
            self.binning_active = False
        else:
            self.binning_active = True
            self.bin_interval = (int)(20000 / 1000) * model_params.bin_time
            self.num_bins = (int)(20000 / self.bin_interval)    

        return
    
    def get_test_data_indices(self):
        all_label_indices = np.arange(len(self.labels))
        to_delete = np.argwhere(np.isin(all_label_indices, self.data_label_indices))
        remaining_label_indices = np.delete(all_label_indices, to_delete)
        return remaining_label_indices
    
    def __len__(self):
        return len(self.data_label_indices)
    
    def _choose_training_dataset(self, num_classes=0):
        unique_labels, label_cnts = np.unique(self.labels, return_counts=True)
        total_num_labels = len(unique_labels)
        
        if (num_classes > 0 and num_classes > total_num_labels) or num_classes <= 0:
            num_classes = total_num_lables
        
        # Get the amount of test data to use for training per class
        data_sample_counts = np.round(label_cnts * self.model_params.train_split).astype(int)
        
        # Make a shuffled selection of indices for test data
        train_indices_per_class = generate_unique_random_arrays(label_cnts, data_sample_counts)
        
        # Offset indexing by the amount of labels per class to properly index original labels data array
        label_index_offset = np.cumsum(np.concatenate([[0], label_cnts[:-1]]))
        
        # Make an array of the label indices for the training dataset
        self.data_label_indices = np.concatenate(add_list_np_arrays_to_offset(train_indices_per_class, label_index_offset))

    # Return the sampling rate (Hz) of the data - can return rate modified from original data sampling rate if data was binned
    def get_sampling_rate(self):
        return self.num_bins
        
    def __getitem__(self, idx):
        file_index = self.data_label_indices[idx]
        file_path = self.data_file_names[file_index]
        raw_sample = pd.read_csv(file_path, skiprows=13, usecols= lambda col: col != "TIME" and col != "CH2").values
        label = self.labels[file_index]

        if self.binning_active:
            binned_raw_sample = np.zeros(self.num_bins)
            for i in range(self.num_bins):
                interval = raw_sample[i*self.bin_interval:(i+1)*self.bin_interval]
                interval_max = np.max(interval)
                interval_min = np.min(interval)
                binned_raw_sample[i] = np.mean(interval)

            # Normalize
            binned_raw_sample = (binned_raw_sample - np.min(binned_raw_sample)) / (np.max(binned_raw_sample) - np.min(binned_raw_sample))
            return binned_raw_sample, label
        else:
            return raw_sample, label


class ModelRunParams():
    def __init__(self):
        self.presentation_time = None
        self.lmu_time_per_sample = None
        self.lmu_total_train_time = None
        self.train_split = None
        self.num_classes = None
        self.epochs = None
        self.sim_time = None
        self.train_time = None
        self.train_time_start = None
        self.test_time = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.batch_size = None
        self.index_at_train_time_end = None
        self.num_fft_channels = None
        self.sampling_rate = None
        self.seed = None
        self.bin_time = None # in milliseconds

    def __str__(self):
        return (
            f"Presentation Time: {self.presentation_time}\n"
            f"LMU Time per Sample: {self.lmu_time_per_sample}\n"
            f"LMU Total Train Time: {self.lmu_total_train_time}\n"
            f"Training Split: {self.train_split}\n"
            f"Number of Classes: {self.num_classes}\n"
            f"Epochs: {self.epochs}\n"
            f"Simulation Time: {self.sim_time}\n"
            f"Training Time: {self.train_time}\n"
            f"Testing Time: {self.test_time}\n"
            f"Test Time Start: {self.test_time_start}\n"
            f"Train Batch Size: {self.train_batch_size}\n"
            f"Test Batch Size: {self.test_batch_size}\n"
            f"Batch Size: {self.batch_size}\n"
            f"Index at Train Time End: {self.index_at_train_time_end}\n"
            f"Number of FFT Channels: {self.num_fft_channels}\n"
            f"Sampling Rate: {self.sampling_rate}\n"
            f"Seed: {self.seed}\n"
            f"Bin Time: {self.bin_time}"
        )

MODEL_PARAMS = ModelRunParams()
MODEL_PARAMS.presentation_time = 2 # in seconds
MODEL_PARAMS.lmu_time_per_sample = 5
MODEL_PARAMS.train_split = 0.8
#MODEL_PARAMS.seed = 10
MODEL_PARAMS.num_classes = 2
MODEL_PARAMS.epochs = 1
MODEL_PARAMS.bin_time = 1 # in milliseconds

"""
Setup
-----------
"""
# Specify directory where an output directory will be created.
project_dir = os.path.expanduser("~/tbp/results/monty/projects")

raw_sensor_train = RawTouchSensorDataset(default_raw_touch_data_dir, MODEL_PARAMS)
raw_sensor_test = RawTouchSensorDataset(default_raw_touch_data_dir, MODEL_PARAMS, raw_sensor_train.get_test_data_indices())

train_data_loader = DataLoader(raw_sensor_train, batch_size=len(raw_sensor_train), shuffle=True, drop_last=True)
train_data, train_targets = next(iter(train_data_loader))

test_data_loader = DataLoader(raw_sensor_test, batch_size=len(raw_sensor_test), shuffle=True, drop_last=True)
test_data, test_targets = next(iter(test_data_loader))

"""
Training
----------------------------------------------------------------------------------------
"""
voltage_touch_test = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        n_train_epochs=2,
        do_eval=False,
        max_train_steps=1001,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=project_dir,
        run_name="voltage_touch_test",
        wandb_handlers=[],
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=1000),
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
        env_init_args={'dataset': {
            'train_data' : train_data,
            'train_targets' : train_targets,
            'test_data' : test_data,
            'test_targets' : test_targets
        }},
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