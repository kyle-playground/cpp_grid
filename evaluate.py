from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from gridworld import CoverageEnv
import time
DEFAULT_OPTIONS = {
    'world_shape': [24, 24],
    'FOV': [5, 5],
    'termination_no_new_coverage': 10,
    'max_episode_len': -1,
    "map_mode": "random",
    "n_agents": 3,
}

env = CoverageEnv(DEFAULT_OPTIONS)
env.reset()
for _ in range(100):
    actions = {
        "agent_0": np.random.randint(low=0, high=5),
        "agent_1": np.random.randint(low=0, high=5),
        "agent_2": np.random.randint(low=0, high=5),
    }
    env.step(actions)
    env.render()