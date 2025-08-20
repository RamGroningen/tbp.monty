import numpy as np
from tbp.monty.frameworks.environments.embodied_environment import EmbodiedEnvironment, ActionSpace
from tbp.monty.frameworks.actions.actions import Action
from scipy.spatial.transform import Rotation

'''Environment Classes'''
class VoltageTouchActionSpace(tuple, ActionSpace):
    """Simple action space for voltage touch environment."""
    def sample(self):
        return self[0] if self else None

class VoltageTouchEnvironment(EmbodiedEnvironment):
    def __init__(self, dataset=None):
        super().__init__()
        self.current_time = 0.0
        self.sensor_location = np.array([0.0, 0.0, 0.1]) # fixed sensor location
        self.voltage_baseline = 0.05 # Baseline voltage
        self.voltage_amplitude = 0.3 # Max voltage when touching
        self.touch_frequency = 2.0 # Hz of touch events
        # Just for compatibility with the dataloader system
        self._agents = [type("FakeAgent", (object,), {"action_space_type": "surface_agent"})()]
        self.fabric_num = -1 # fabric to get data from - first run increments this pre-data access so start at -1

        if isinstance(dataset, dict):
            self.train_data = dataset['train_data'] if 'train_data' in dataset else None
            self.train_targets = dataset['train_targets'] if 'train_targets' in dataset else None
            self.test_data = dataset['test_data'] if 'test_data' in dataset else None
            self.test_targets = dataset['test_targets'] if 'test_targets' in dataset else None

        self.fabric_rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
        self.rotation_step = 360/1000

        # TODO - if incorrect dataset type fails silently - can throw an error to help later

    @property
    def action_space(self):
        """Return the action space for this environment."""
        # For a simple voltage touch environment, we might not need complex actions
        # You can customize this based on what actions you want to support
        return VoltageTouchActionSpace(["no_action"])
    
    def add_object(self, name, position = ..., rotation = ..., scale = ..., semantic_id = None, enable_physics = False, object_to_avoid=False, primary_target_object=None):
        pass

    def remove_all_objects(self):
        pass

    def pre_epoch(self):
        self.fabric_num += 1

    def close(self):
        """Close the environment and release resources."""
        # Clean up any resources if needed
        pass

    def step(self, action: Action) -> dict:
        """Simulate one step of the environment, generating voltage data."""
        self.current_time += 0.001 # 100ms time steps
        # Simulate voltage data based on time
        # This creates a relatistic pattern of touch events
        idx = int(self.current_time / 0.001)
        voltage = self._generate_voltage_data(idx)
        self.fabric_rotation = self.fabric_rotation * Rotation.from_euler('z', self.rotation_step, degrees=True)

        # Create observation with voltage data
        obs = {
            "agent_id_0": {
                "finger": {
                    "voltage" : voltage,
                    "fabric_id" : self.train_targets[self.fabric_num],
                    "timestamp" : self.current_time,
                    "location" : self.sensor_location,
                }
            }
        }

        return obs

    def _generate_voltage_data(self, idx):
        out = self.train_data[self.fabric_num][idx]
        return out.item()
        
        """Generate voltage data"""
        # Add som enoise to baseline
        voltage = self.voltage_baseline + np.random.normal(0, 0.01)

        # Add periodic touch events
        touch_phase = (self.current_time * self.touch_frequency) % 1.0
        if touch_phase < 0.2: # touch for 20% of the cycle
            if touch_phase < 0.1:
                # Pressing down - voltage increases
                voltage += self.voltage_amplitude * (touch_phase / 0.1)
            else:
                # Releasing - voltage decreases
                voltage -= self.voltage_amplitude * (1.0 - (touch_phase - 0.1) / 0.1)

        # Add some random touch events (10% chance per step)
        if np.random.random() < 0.1:
            voltage += self.voltage_amplitude * np.random.random()

        return np.clip(voltage, 0.0, 1.0)

    def get_state(self):
        """Get agent state with sensor information."""
        state = {
            "agent_id_0": {
                "finger": {  # This should match your sensor module ID
                    "rotation": np.array([1.0, 0.0, 0.0, 0.0]), # Identity quaternion
                    "position": self.sensor_location,
                },
                "rotation": np.array([1.0, 0.0, 0.0, 0.0]),
                "position": np.array([0.0, 0.0, 0.0]),
            }
        }
        return state

    def reset(self):
        """Reset the environment"""
        self.current_time = 0.0
        return self.get_state()