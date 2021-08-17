from skimage.draw import random_shapes
from skimage.measure import label
from scipy import ndimage
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib import colors
import gym
import copy
from gym import spaces
from gym.utils import seeding, EzPickle
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID


DEFAULT_OPTIONS = {
    'world_shape': [24, 24],
    'FOV': [5, 5],
    'termination_no_new_coverage': 10,
    'max_episode_len': -1,
    "map_mode": "random",
    "n_agents": 3,
}

X = 1
Y = 0


class GridWorld(object):
    def __init__(self, shape):
        # Grid world
        self.shape = tuple(shape)  # why tuple?
        self.map = None
        self.world = None
        self.coverage = None
        self.all_coverable_area = None
        self.explored_map = np.zeros(self.shape, dtype=int)
        self.reset()

    def reset(self):
        # Every agent start at origin every time
        self.coverage = np.zeros(self.shape, dtype=np.int)
        self.map = self._random_obstacles_map(width=self.shape[0], height=self.shape[0], max_shapes=6, min_shapes=6,
                                              max_size=5, min_size=4, allow_overlap=True, shape='ellipse')
        self.world = copy.deepcopy(self.map)
        self.all_coverable_area = self.get_coverable_area()

    def get_coverable_area_faction(self):
        coverable_area = ~(self.map > 0)
        return np.sum(coverable_area)/(self.map.shape[X]*self.map.shape[Y])

    def get_coverable_area(self):
        coverable_area = ~(self.map > 0)
        return np.sum(coverable_area)

    def get_covered_area(self):
        coverable_area = ~(self.map > 0)
        return np.sum((self.coverage > 0) & coverable_area)

    def get_coverage_fraction(self):
        coverable_area = ~(self.map > 0)
        covered_area = (self.coverage > 0) & coverable_area
        return np.sum(covered_area)/np.sum(coverable_area)

    @staticmethod
    def _random_obstacles_map(width, height, max_shapes, min_shapes, max_size, min_size,
                              allow_overlap, shape=None):
        def build_map():
            x, _ = random_shapes((height, width), max_shapes=max_shapes, min_shapes=min_shapes,
                                max_size=max_size, min_size=min_size, multichannel=False, shape=shape,
                                allow_overlap=allow_overlap,random_seed=1)
            x[x == 255] = 0
            x[np.nonzero(x)] = 1
            x = ndimage.binary_fill_holes(x).astype(int)
            return x
        x = build_map()
        while (label(~(x>0)) > 1).any():
            x = build_map()
        return x

class Action(Enum):
    FREEZE     = 0
    MOVE_RIGHT = 1
    MOVE_LEFT  = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

class Explorer(object):
    def __init__(self,
                 random_state,  # same number generator.
                 agent_index,
                 gridworld,
                 fov,
                 ):

        self.agent_id = agent_index
        self.gridworld = gridworld
        self.FOV = fov
        self.state = None
        self.position = None
        self.prev_pos = None
        self.coverage = None
        self.local_map = None
        self.revisit = 0
        self.no_new_coverage_steps = None
        self.termination_no_new_coverage = 10
        self.reward = None
        self.reset(random_state)

    def reset(self, random_state):
        def random_pos(pos_mean ,var):
            return np.array([
                int(np.clip(random_state.normal(loc=pos_mean[c], scale=var), 0, self.gridworld.map.shape[c] - 1))
                for c in [Y, X]])

        pos_mean = np.array([random_state.randint(0, self.gridworld.map.shape[c]) for c in [Y, X]])
        var = 1
        self.position = random_pos(pos_mean, var)
        while self.gridworld.map.map[self.position[Y], self.position[X]] == 1 or self.gridworld.is_occupied(self.position, self):
            var += 0.2
            self.position = random_pos(pos_mean, var)

        self.coverage = np.zeros(self.gridworld.map.shape, dtype=np.bool)
        self.local_map = np.zeros(self.gridworld.map.shape, dtype=int)
        self.no_new_coverage_steps = 0
        self.reward = 0
        self.revisit = 0

    def step(self,action):
        action = Action(action)  # action = Action.Move

        delta_pos = {
            Action.MOVE_RIGHT:  [ 0,  1],
            Action.MOVE_LEFT:   [ 0, -1],
            Action.MOVE_UP:     [-1,  0],
            Action.MOVE_DOWN:   [ 1,  0],
            Action.FREEZE:      [ 0,  0]
        }[action]  # Convert 0...4 to Action.MOVE then to operation [? , ?]

        is_valid_pos = lambda p: all([p[c] >= 0 and p[c] < self.gridworld.map.shape[c] for c in [Y, X]])  # All return ture if all elments are true
        is_obstacle = lambda p: self.gridworld.map.map[p[Y]][p[X]] == 1

        self.prev_pos = self.position
        desired_pos = self.position + delta_pos
        if is_valid_pos(desired_pos) and not is_obstacle(desired_pos) and not self.gridworld.is_occupied(desired_pos, self):
            self.position = desired_pos

        if self.gridworld.map.coverage[self.position[Y], self.position[X]] == 0:
            self.gridworld.map.coverage[self.position[Y], self.position[X]] = self.agent_id
            self.reward = 1
            self.no_new_coverage_steps = 0
        else:
            self.reward = 0
            self.revisit += 1
            self.no_new_coverage_steps += 1

        self.coverage[self.position[Y], self.position[X]] = True

    def update_state(self):
        coverage = self.coverage.copy().astype(np.int)
        clip_boundary = lambda b: np.clip(b, [0,0], self.gridworld.map.shape)
        bound_l = clip_boundary(self.position - (self.FOV - 1) / 2)
        bound_u = clip_boundary(self.position + (self.FOV - 1) / 2) + 1

        self.local_map[int(bound_l[Y]):int(bound_u[Y]), int(bound_l[X]):int(bound_u[X])] = self.gridworld.map.world[int(bound_l[Y]):int(bound_u[Y]), int(bound_l[X]):int(bound_u[X])]

        state_data = [
            self.local_map,
            coverage,
        ]
        self.state = np.stack(state_data, axis=-1).astype(np.int)
        done = self.no_new_coverage_steps == self.termination_no_new_coverage
        info = self.revisit

        return self.state, self.reward, done, info


class CoverageEnv(MultiAgentEnv):
    """
    single_agent_observation_space = spaces.Tuple(
                [spaces.Box(0, np.inf, shape=DEFAULT_OPTIONS['world_shape'] + [2*DEFAULT_OPTIONS['n_agents']], dtype=np.int64),
                 spaces.Box(low=np.array([0, 0, 0] * DEFAULT_OPTIONS['n_agents']),
                            high=np.array([DEFAULT_OPTIONS['world_shape'][Y], DEFAULT_OPTIONS['world_shape'][X], 1] * DEFAULT_OPTIONS['n_agents']), dtype=np.int64),
                 spaces.Box(0, np.inf, shape=DEFAULT_OPTIONS['world_shape']+[3], dtype=np.int64),
                 ])
    """
    single_agent_observation_space = spaces.Tuple(
                [spaces.Box(-1, np.inf, shape=DEFAULT_OPTIONS['world_shape'] + [2*DEFAULT_OPTIONS['n_agents']]),
                 spaces.Box(low=np.array([-1, -1, -1] * DEFAULT_OPTIONS['n_agents']),
                            high=np.array([DEFAULT_OPTIONS['world_shape'][Y], DEFAULT_OPTIONS['world_shape'][X], 2] * DEFAULT_OPTIONS['n_agents'])),
                 spaces.Box(-1, np.inf, shape=DEFAULT_OPTIONS['world_shape']+[3]),
                 ])
    single_agent_action_space = spaces.Discrete(5)

    def __init__(self, env_config):
        # EzPickle.__init__(self)
        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(env_config)
        self.agent_random_state, seed_agents = seeding.np_random()
        self.map = GridWorld(self.cfg['world_shape'])
        self.team = []
        for i in range(self.cfg['n_agents']):
            self.team.append(Explorer(self.agent_random_state, i, self, np.array(self.cfg['FOV'])))
        self.timestep = None

        self.observation_space = spaces.Tuple(
                [spaces.Box(-1, np.inf, shape=self.cfg['world_shape'] + [2*self.cfg['n_agents']]),
                 spaces.Box(low=np.array([-1, -1, -1] * self.cfg['n_agents']),
                            high=np.array([self.cfg['world_shape'][Y], self.cfg['world_shape'][X], 2] * self.cfg['n_agents'])),
                 spaces.Box(-1, np.inf, shape=self.cfg['world_shape']+[3]),
                 ])
        self.action_space = spaces.Discrete(5)

        # TODO: set color for rendering
        self.map_colormap = colors.ListedColormap(['white', 'black', 'gray'])
        self.team_agents_color = colors.hsv_to_rgb(np.linspace(160 / 360, 250 / 360, self.cfg['n_agents']))

    def is_occupied(self, p, agent_ignore=None):
        for o in self.team:
            if o is agent_ignore:
                continue
            if p[X] == o.position[X] and p[Y] == o.position[Y]:
                return True
        return False

    def reset(self):
        self.timestep = 0
        self.map.reset()
        for agent in self.team:
            agent.reset(self.agent_random_state)

        freeze_all_agents = {
            "agent_0": Action.FREEZE,
            "agent_1": Action.FREEZE,
            "agent_2": Action.FREEZE,
        }
        return self.step(freeze_all_agents)[0]  # [0] -> state(observation)

    def step(self, action_dict):
        def mark_self_obs(own_state):
            for i in range(len(own_state)):
                own_state[i] += own_state[i]
            return state

        self.timestep += 1

        action_index = 0
        for agent in self.team:
            agent.step(action_dict["agent_{}".format(action_index)])
            action_index += 1

        states, dones, rewards, revisits = [], [], [], []
        total_rewards = 0
        total_revisit = 0

        for i, agent in enumerate(self.team):
            state, reward, done, info = agent.update_state()
            states.append(state)
            rewards.append(reward)
            total_rewards += reward
            total_revisit += info
            dones.append(done)

        world_terminator = self.timestep == (self.map.all_coverable_area+10) or self.map.get_coverage_fraction() == 1.0
        all_done = all(dones) or world_terminator

        agents_pos_map = np.zeros(self.map.shape, dtype=np.uint8)
        for agent in self.team:
            agents_pos_map[agent.position[Y], agent.position[X]] = 1
        global_state = np.stack([self.map.map, self.map.coverage > 0, agents_pos_map], axis=-1)
        # use axis=-1 (because tensor(batch, width, hight, channel)
        state = {
            'agent_0': tuple(
                [np.concatenate((mark_self_obs(states[0]),
                                 states[1],
                                 states[2]), axis=-1),
                 np.concatenate((np.append(self.team[0].position, 1),
                                 np.append(self.team[1].position, 0),
                                 np.append(self.team[2].position, 0)), axis=0),
                 global_state,
                 ]),
            'agent_1': tuple(
                [np.concatenate((mark_self_obs(states[1]),
                                 states[0],
                                 states[2]), axis=-1),
                 np.concatenate((np.append(self.team[1].position, 1),
                                 np.append(self.team[0].position, 0),
                                 np.append(self.team[2].position, 0)), axis=0),
                 global_state,
                 ]),
            'agent_2': tuple(
                [np.concatenate((mark_self_obs(states[2]),
                                 states[0],
                                 states[1]), axis=-1),
                 np.concatenate((np.append(self.team[2].position, 1),
                                 np.append(self.team[0].position, 0),
                                 np.append(self.team[1].position, 0)), axis=0),
                 global_state,
                 ]),
        }
        """
        state_all_image = {
            'agent_0': tuple(
                [np.concatenate((mark_self_obs(states[0]),
                                 states[1],
                                 states[2]), axis=-1),
                 global_state,
                 ]),
            'agent_1': tuple(
                [np.concatenate((mark_self_obs(states[1]),
                                 states[0],
                                 states[2]), axis=-1),
                 global_state,
                 ]),
            'agent_2': tuple(
                [np.concatenate((mark_self_obs(states[2]),
                                 states[0],
                                 states[1]), axis=-1),
                 global_state,
                 ]),
        }
        """
        reward = {
            'agent_0': total_rewards / 3.0,
            'agent_1': total_rewards / 3.0,
            'agent_2': total_rewards / 3.0,
        }
        done = {"__all__": all_done}
        info = {
            'agent_0': {
                'current_global_coverage': self.map.get_coverage_fraction(),
                'coverable_area': self.map.get_coverable_area(),
                'rewards_team': total_rewards,
                'revisit steps': total_revisit,
            },
            'agent_1': {
                'current_global_coverage': self.map.get_coverage_fraction(),
                'coverable_area': self.map.get_coverable_area(),
                'rewards_team': total_rewards,
                'revisit steps': total_revisit,
            },
            'agent_2': {
                'current_global_coverage': self.map.get_coverage_fraction(),
                'coverable_area': self.map.get_coverable_area(),
                'rewards_team': total_rewards,
                'revisit steps': total_revisit,
            },
        }
        """
        'current_global_coverage': self.map.get_coverage_fraction(),
        'coverable_area': self.map.get_coverable_area(),
        'rewards_team': total_rewards,
        'revisit steps': total_revisit,
        """
        del states, dones, rewards, revisits

        return state, reward, done, info
