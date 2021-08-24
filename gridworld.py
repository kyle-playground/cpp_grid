from skimage.draw import random_shapes
from skimage.measure import label
from scipy import ndimage
import copy
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib import colors

import gym
from gym import spaces
from gym.utils import seeding

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# remote server cannot use TKAgg backend comment out while training
# matplotlib.use("TkAgg")

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
            # TODO:  erase random seed = 1, not it's just for examination varifying policy does converge
            x, _ = random_shapes((height, width), max_shapes=max_shapes, min_shapes=min_shapes,
                                max_size=max_size, min_size=min_size, multichannel=False, shape=shape,
                                allow_overlap=allow_overlap, random_seed=1)
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
        self.step_reward = None
        self.total_reward = 0
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
        self.local_map = np.ones(self.gridworld.map.shape, dtype=int) * 2
        self.no_new_coverage_steps = 0
        self.step_reward = 0
        self.revisit = 0
        self.total_reward = 0

    def step(self,action):
        action = Action(action)  # action = Action.Move

        delta_pos = {
            Action.MOVE_RIGHT:  [ 0,  1],
            Action.MOVE_LEFT:   [ 0, -1],
            Action.MOVE_UP:     [-1,  0],
            Action.MOVE_DOWN:   [ 1,  0],
            Action.FREEZE:      [ 0,  0],
        }[action]  # Convert 0...4 to Action.MOVE then to operation [? , ?]

        is_valid_pos = lambda p: all([p[c] >= 0 and p[c] < self.gridworld.map.shape[c] for c in [Y, X]])  # All return ture if all elments are true
        is_obstacle = lambda p: self.gridworld.map.map[p[Y]][p[X]] == 1

        self.prev_pos = self.position.copy()
        desired_pos = self.position + delta_pos
        if is_valid_pos(desired_pos) and not is_obstacle(desired_pos) and not self.gridworld.is_occupied(desired_pos, self):
            self.position = desired_pos

        if self.gridworld.map.coverage[self.position[Y], self.position[X]] == 0:
            self.gridworld.map.coverage[self.position[Y], self.position[X]] = self.agent_id
            self.step_reward = 1
            self.no_new_coverage_steps = 0
        else:
            self.step_reward = -0.2
            self.revisit += 1
            self.no_new_coverage_steps += 1

        self.total_reward += self.step_reward

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

        return self.state, self.step_reward, done, info


class CoverageEnv(MultiAgentEnv):
    # TODO: take revisited areas in consideration
    single_agent_observation_space = spaces.Tuple(
                [spaces.Box(np.float32(-1), np.float32(5), shape=DEFAULT_OPTIONS['world_shape'] + [2]),
                 spaces.Box(np.float32(-1), np.float32(5), shape=DEFAULT_OPTIONS['world_shape'] + [2]),
                 spaces.Box(np.float32(-1), np.float32(5), shape=DEFAULT_OPTIONS['world_shape'] + [2]),
                 spaces.Box(low=np.array([-1, -1, -1] * DEFAULT_OPTIONS['n_agents']),
                            high=np.array([DEFAULT_OPTIONS['world_shape'][Y], DEFAULT_OPTIONS['world_shape'][X], 2] * DEFAULT_OPTIONS['n_agents'])),
                 spaces.Box(np.float32(-1), np.float32(5), shape=DEFAULT_OPTIONS['world_shape']+[3]),
                 ])
    single_agent_action_space = spaces.Discrete(5)

    def __init__(self, env_config=DEFAULT_OPTIONS):
        # EzPickle.__init__(self)
        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(env_config)
        self.agent_random_state, seed_agents = seeding.np_random()
        self.map = GridWorld(self.cfg['world_shape'])
        self.team = []
        for i in range(self.cfg['n_agents']):
            self.team.append(Explorer(self.agent_random_state, i+1, self, np.array(self.cfg['FOV'])))
        self.timestep = None
        self.termination = None
        # Performance evaluation
        self.total_reward = 0

        self.observation_space = spaces.Tuple(
                [spaces.Box(np.float32(-1), np.float32(5), shape=self.cfg['world_shape'] + [2]),
                 spaces.Box(np.float32(-1), np.float32(5), shape=self.cfg['world_shape'] + [2]),
                 spaces.Box(np.float32(-1), np.float32(5), shape=self.cfg['world_shape'] + [2]),
                 spaces.Box(low=np.array([-1, -1, -1] * self.cfg['n_agents']),
                            high=np.array([self.cfg['world_shape'][Y], self.cfg['world_shape'][X], 2] * self.cfg['n_agents'])),
                 spaces.Box(np.float32(-1), np.float32(5), shape=self.cfg['world_shape']+[3]),
                 ])
        self.action_space = spaces.Discrete(5)

        # set color for rendering env
        self.fig = None
        self.map_colormap = colors.ListedColormap(['white', 'black', 'gray'])
        hsv = np.ones((self.cfg['n_agents'], 3))
        hsv[..., 0] = np.linspace(160 / 360, 250 / 360, self.cfg['n_agents'])
        self.team_agents_color = colors.hsv_to_rgb(hsv)

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
        self.termination = int(self.map.all_coverable_area / 3) + 10
        self.total_reward = 0
        for agent in self.team:
            agent.reset(self.agent_random_state)

        freeze_all_agents = {
            "agent_0": Action.FREEZE,
            "agent_1": Action.FREEZE,
            "agent_2": Action.FREEZE,
        }
        return self.step(freeze_all_agents)[0]  # [0] -> state(observation)

    def step(self, action_dict):
        # def mark_self_obs(own_state):
        #     for i in range(len(own_state)):
        #         own_state[i] += own_state[i]
        #     return own_state

        self.timestep += 1

        action_index = 0
        for agent in self.team:
            agent.step(action_dict["agent_{}".format(action_index)])
            action_index += 1

        states, dones, rewards, revisits = [], [], [], []
        total_rewards_per_step = 0
        for i, agent in enumerate(self.team):
            state, reward, done, info = agent.update_state()
            states.append(state)
            rewards.append(reward)
            total_rewards_per_step += reward
            revisits.append(info)
            dones.append(done)
        self.total_reward += total_rewards_per_step

        world_terminator = self.timestep == self.termination or self.map.get_coverage_fraction() == 1.0
        # all_done = all(dones) or world_terminator
        all_done = world_terminator

        agents_pos_map = np.zeros(self.map.shape, dtype=np.uint8)
        for agent in self.team:
            agents_pos_map[agent.position[Y], agent.position[X]] = agent.agent_id
        global_state = np.stack([self.map.map, self.map.coverage > 0, agents_pos_map], axis=-1)
        # use axis=-1 (because tensor(batch, width, hight, channel)
        state = {
            'agent_0': tuple(
                [states[0],
                 states[1],
                 states[2],
                 np.concatenate((np.append(self.team[0].position, 1),
                                 np.append(self.team[1].position, 0),
                                 np.append(self.team[2].position, 0)), axis=0),
                 global_state,
                 ]),
            'agent_1': tuple(
                [states[1],
                 states[0],
                 states[2],
                 np.concatenate((np.append(self.team[1].position, 1),
                                 np.append(self.team[0].position, 0),
                                 np.append(self.team[2].position, 0)), axis=0),
                 global_state,
                 ]),
            'agent_2': tuple(
                [states[2],
                 states[0],
                 states[1],
                 np.concatenate((np.append(self.team[2].position, 1),
                                 np.append(self.team[0].position, 0),
                                 np.append(self.team[1].position, 0)), axis=0),
                 global_state,
                 ]),
        }
        reward = {
            'agent_0': total_rewards_per_step / 3.0,
            'agent_1': total_rewards_per_step / 3.0,
            'agent_2': total_rewards_per_step / 3.0,
        }
        done = {"__all__": all_done}
        info = {
            'agent_0': {
                'all_coverable_area': self.map.get_coverable_area(),
                'current_global_coverage': self.map.get_coverage_fraction(),
                'self_reward': self.team[0].total_reward,
                'rewards_team': self.total_reward,
                'revisit steps': revisits[0],
            },
            'agent_1': {
                'all_coverable_area': self.map.get_coverable_area(),
                'current_global_coverage': self.map.get_coverage_fraction(),
                'self_reward': self.team[1].total_reward,
                'rewards_team': self.total_reward,
                'revisit steps': revisits[1],
            },
            'agent_2': {
                'all_coverable_area': self.map.get_coverable_area(),
                'current_global_coverage_fraction': self.map.get_coverage_fraction(),
                'self_reward': self.team[2].total_reward,
                'rewards_team': self.total_reward,
                'revisit steps': revisits[2],
            },
        }
        del states, dones, rewards, revisits

        return state, reward, done, info

    def clear_patches(self, ax):
        [p.remove() for p in reversed(ax.patches)]
        [t.remove() for t in reversed(ax.texts)]

    def render_overview(self, ax, stepsize=1.0):
        if not hasattr(self, 'im_map'):
            ax.set_xticks([])
            ax.set_yticks([])
            self.im_map = ax.imshow(np.zeros(self.map.shape), vmin=0, vmax=3)

        # Turn map into white, black, and gray
        self.im_map.set_data(self.map_colormap(self.map.map))
        for _, agent in enumerate(self.team):
            rect_size = 1
            pose_microstep = agent.prev_pos + (agent.position - agent.prev_pos) * stepsize
            rect = patches.Rectangle((pose_microstep[1] - rect_size / 2, pose_microstep[0] - rect_size / 2), rect_size,
                                     rect_size,
                                     linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

    def render_global_coverages(self, ax):
        if not hasattr(self, 'im_cov_global'):
            self.im_cov_global = ax.imshow(np.zeros(self.map.shape), vmin=0, vmax=100)
        all_team_colors = [(0, 0, 0, 0)] + [tuple(list(c) + [0.5]) for c in self.team_agents_color]
        coverage = self.map.coverage.copy()
        self.im_cov_global.set_data(colors.ListedColormap(all_team_colors)(coverage))

    def render(self, mode=None, stepsize=1.0):
        if self.fig is None:
            # interaction mode on
            plt.ion()
            # set figure size (inch x inch)
            self.fig = plt.figure(figsize=(5, 5))
            # subplot 111 means one graph
            self.ax_overview = self.fig.add_subplot(1, 1, 1, aspect='equal')
        # clear patch for next image
        self.clear_patches(self.ax_overview)
        # add agents' patches in ax
        self.render_overview(self.ax_overview, stepsize)
        # add coverage
        self.render_global_coverages(self.ax_overview)
        # self.fig.canvas.draw()
        plt.draw()
        plt.pause(0.001)
        return self.fig
