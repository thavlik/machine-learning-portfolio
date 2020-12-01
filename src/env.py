import numpy as np
from torch.nn import functional as F
from gym import Env, spaces
from dataset import get_dataset
from merge_strategy import strategy


class TimeSeriesDetector(Env):
    def __init__(self, config: dict):
        super(TimeSeriesDetector, self).__init__()
        self.observation_length = config['observation_length']
        self.num_channels = config['channels']
        self.action_stride = config.get('action_stride', 0)
        num_actions = config['num_event_classes']+1  # abstain
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(
            low=config['low'],
            high=config['high'],
            shape=(self.num_channels, self.observation_length),
            dtype=np.float32,
        )
        # Load the data
        self.ds = get_dataset(**config['data'])

    def step(self, action: int):
        reward = 0.0
        y = self.y[:, self.current_step:self.current_step +
                   self.observation_length]
        #loss = F.nll_loss(prediction, target)
            
        self.current_step += 1
        if action != 0:
            self.current_step += self.action_stride
        obs = self.get_observation()
        done = self.current_step >= self.x.shape[1] - self.observation_length
        info = self.get_info_dict()
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        i = np.random.randint(0, len(self.ds))
        self.x, self.y = self.ds[i]
        if self.x.shape[1] < self.observation_length:
            raise ValueError(f'Example {i} is shorter ({self.x.shape[1]}) '
                             f'than the observation length ({self.observation_length})')
        return self.get_observation()

    def get_observation(self):
        return self.x[:, self.current_step:self.current_step +
                      self.observation_length].numpy()

    def get_info_dict(self) -> dict:
        return dict(current_step=self.current_step)


envs = {
    'TimeSeriesDetector': TimeSeriesDetector,
}


def get_env(name: str):
    if name not in envs:
        raise ValueError(f'Environment "{name}" not found '
                         f'valid options are {envs}')
    return envs[name]
