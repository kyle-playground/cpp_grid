from skimage.draw import random_shapes
from skimage.measure import label
from scipy import ndimage
import copy
import math
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

from astar import AStar
# remote server cannot use TKAgg backend comment out while training
# matplotlib.use("TkAgg")

DEFAULT_OPTIONS = {
    'world_shape': [14, 14],
    "state_size": 28,
    'FOV': [5, 5],
    'termination_no_new_coverage': 10,
    'max_episode_len': -1,
    "map_mode": "known",
    "n_agents": 3,
    "centered_state": False,
    "revisit_penalty": True,
    "agent_distance_penalty": False,
}

X = 1
Y = 0


class GridWorld(object):
    def __init__(self, shape, random_state, min_coverable_area_fraction=0.6):
        # Grid world
        self.shape = tuple(shape)  # why tuple?
        self.width = self.shape[X]
        self.height = self.shape[Y]
        self.map = None
        self.coverage = None
        self.all_coverable_area = None
        self.explored_map = np.zeros(self.shape, dtype=int)
        self.min_coverable_area_fraction = min_coverable_area_fraction
        self.reset(random_state)

    def reset(self, random_state):
        # Every agent start at origin every time
        self.coverage = np.zeros(self.shape, dtype=np.int)
        # self.map = self._random_obstacles_map(width=self.shape[0], height=self.shape[0], max_shapes=4, min_shapes=4,
        #                                       max_size=3, min_size=2, allow_overlap=False, shape='ellipse')
        self.build_map(random_state)
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

    def build_map(self, random_state):
        if self.min_coverable_area_fraction == 1.0:
            self.map = np.zeros(self.shape, dtype=np.uint8)
        else:
            self.map = np.ones(self.shape, dtype=np.uint8)
            p = np.array([random_state.randint(0, self.shape[c]) for c in [Y, X]])
            while self.get_coverable_area_faction() < self.min_coverable_area_fraction:
                d_p = np.array(
                    [[0, 1], [0, -1], [-1, 0], [1, 0]][random_state.randint(0, 4)])  # *random_state.randint(1, 5)
                p_new = np.clip(p + d_p, [0, 0], np.array(self.shape) - 1)
                self.map[min(p[Y], p_new[Y]):max(p[Y], p_new[Y]) + 1, min(p[X], p_new[X]):max(p[X], p_new[X]) + 1] = 0
                p = p_new

    @staticmethod
    def _random_obstacles_map(width, height, max_shapes, min_shapes, max_size, min_size,
                              allow_overlap, shape=None):
        def build_map():

            x, _ = random_shapes((height, width), max_shapes=max_shapes, min_shapes=min_shapes,
                                max_size=max_size, min_size=min_size, multichannel=False, shape=shape,
                                allow_overlap=allow_overlap)
            x[x == 255] = 0
            x[np.nonzero(x)] = 1
            x = ndimage.binary_fill_holes(x).astype(int)
            x[0, ...] = 0

            return x
        x = build_map()
        while (label(~(x>0)) > 1).any():
            x = build_map()
        return x

class MazeSolver(AStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, map):
        self.map = map
        self.width, self.height = map.shape

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        return 1

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node
        return[(nx, ny) for nx, ny in[(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]if 0 <= nx < self.width and 0 <= ny < self.height and self.map[nx][ny] == 0]


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
                 revisit_penalty=False
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
        self.prev_act = None
        self.revisit_penalty = revisit_penalty
        self.reset(random_state)

    def reset(self, random_state):
        def random_pos(pos_mean ,var):
            return np.array([
                int(np.clip(random_state.normal(loc=pos_mean[c], scale=var), 0, self.gridworld.map.shape[c] - 1))
                for c in [Y, X]])
        # self.position = np.array([0,self.agent_id])
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
        self.step_reward = 0
        # self.step_reward += -0.01

        delta_pos = {
            Action.MOVE_RIGHT:  [ 0,  1],
            Action.MOVE_LEFT:   [ 0, -1],
            Action.MOVE_UP:     [-1,  0],
            Action.MOVE_DOWN:   [ 1,  0],
            Action.FREEZE:      [ 0,  0],
        }[action]  # Convert 0...4 to Action.MOVE then to operation [? , ?]

        # Energy cost
        # if self.prev_act == action:
        #     self.step_reward += 0.1
        # self.prev_act = action

        is_valid_pos = lambda p: all([p[c] >= 0 and p[c] < self.gridworld.map.shape[c] for c in [Y, X]])  # All return ture if all elments are true
        is_obstacle = lambda p: self.gridworld.map.map[p[Y]][p[X]] == 1

        self.prev_pos = self.position.copy()
        desired_pos = self.position + delta_pos
        if is_valid_pos(desired_pos) and not is_obstacle(desired_pos) and not self.gridworld.is_occupied(desired_pos, self):
            self.position = desired_pos

        if self.gridworld.map.coverage[self.position[Y], self.position[X]] == 0:
            self.gridworld.map.coverage[self.position[Y], self.position[X]] = self.agent_id
            self.step_reward += 1
            # self.no_new_coverage_steps = 0
        else:
            if self.revisit_penalty:
                self.step_reward = -0.1
            else:
                self.step_reward += 0
            self.revisit += 1
            # self.no_new_coverage_steps += 1

        self.total_reward += self.step_reward

        self.coverage[self.position[Y], self.position[X]] = True

    def update_state(self):
        # coverage = self.coverage.copy().astype(np.int)
        # TODO: uncomment when face unknown area
        # clip_boundary = lambda b: np.clip(b, [0,0], self.gridworld.map.shape)
        # bound_l = clip_boundary(self.position - (self.FOV - 1) / 2)
        # bound_u = clip_boundary(self.position + (self.FOV - 1) / 2) + 1
        #
        # self.local_map[int(bound_l[Y]):int(bound_u[Y]), int(bound_l[X]):int(bound_u[X])] = self.gridworld.map.map[int(bound_l[Y]):int(bound_u[Y]), int(bound_l[X]):int(bound_u[X])]
        #
        # state_data = [
        #     self.local_map,
        #     coverage,
        # ]
        # self.state = np.stack(state_data, axis=-1).astype(np.int)
        self.state = []

        done = self.no_new_coverage_steps == self.termination_no_new_coverage
        info = self.revisit

        return self.state, self.step_reward, done, info

    def to_coordinate_frame(self, m, output_shape, fill=0):
        half_out_shape = np.array(output_shape / 2, dtype=np.int)
        padded = np.pad(m, ([half_out_shape[Y]] * 2, [half_out_shape[X]] * 2), mode='constant', constant_values=fill)
        return padded[self.position[Y]:self.position[Y] + output_shape[Y], self.position[X]:self.position[X] + output_shape[Y]]


class CoverageEnv(MultiAgentEnv):
    single_agent_merge_obs_space = spaces.Tuple(
        [spaces.Box(-10, 10, shape=DEFAULT_OPTIONS["world_shape"] + [4], dtype=np.float32),
         spaces.Box(-10, 10, shape=DEFAULT_OPTIONS['world_shape'] + [4], dtype=np.float32),
         ])
    single_agent_action_space = spaces.Discrete(5)

    def __init__(self, env_config=DEFAULT_OPTIONS):
        self.cfg = {}
        self.cfg.update(env_config)
        self.random_state, seed_agents = seeding.np_random(seed=1)

        self.map = GridWorld(self.cfg['world_shape'], random_state=self.random_state)
        self.mazesolver = None  # A star
        # Agents
        self.team = []
        for i in range(self.cfg['n_agents']):
            self.team.append(Explorer(self.random_state,
                                      i+1,
                                      self,
                                      np.array(self.cfg['FOV']),
                                      revisit_penalty=self.cfg["revisit_penalty"]))
        self.timestep = None
        self.termination = None
        # Options
        self.centering = self.cfg['centered_state']
        self.too_close_penalty = self.cfg["agent_distance_penalty"]
        # Performance evaluation
        self.total_reward = 0

        self.observation_space = spaces.Tuple(
            [spaces.Box(-10, 10, shape=self.cfg["world_shape"] + [4], dtype=np.float32),
             spaces.Box(-10, 10, shape=self.cfg['world_shape'] + [4], dtype=np.float32),
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
        self.map.reset(self.random_state)
        self.mazesolver = MazeSolver(self.map.map)
        # self.termination = int(self.map.all_coverable_area / 3)
        self.termination = 50

        self.total_reward = 0
        for agent in self.team:
            agent.reset(self.random_state)
        freeze_all_agents = {"agent_{}".format(i): Action.FREEZE for i in range(self.cfg["n_agents"])}

        return self.step(freeze_all_agents)[0]  # [0] -> state(observation)

    def step(self, action_dict):
        self.timestep += 1
        action_index = 0
        for agent in self.team:
            agent.step(action_dict["agent_{}".format(action_index)])
            action_index += 1

        states, dones, rewards, revisits = [], [], [], []
        total_rewards_per_step = 0
        # if map_model = 'unknown'
        # local_map_merge = np.ones(self.map.shape) * 2
        # Step update
        for i, agent in enumerate(self.team):
            state, reward, done, info = agent.update_state()
            # Merge operation for unknown environment (keep)
            # if self.merge:
            #     local_map = state[..., 0]
            #     local_known_area = (local_map < 2)
            #     known_last_merge = (local_map_merge < 2)
            #     known_area = known_last_merge.astype(np.int) + local_known_area.astype(np.int)
            #     unknown_merge = (known_area == 0)
            #     local_map[local_map == 2] = 0
            #     local_map_merge[local_map_merge == 2] = 0
            #     known_area_merge = (local_map_merge.astype(np.bool) | local_map.astype(np.bool))
            #     local_map_merge = known_area_merge.astype(np.int) + unknown_merge.astype(np.int) * 2
            # else:
            #     states.append(state)
            rewards.append(reward)
            total_rewards_per_step += reward
            revisits.append(info)
            dones.append(done)
        self.total_reward += total_rewards_per_step
        # Too-close penalty
        radius = 3
        if self.too_close_penalty:
            position_list = [agent.position for _, agent in enumerate(self.team)]
            for i in range(len(position_list) - 1):
                for j in range(len(position_list) - 1 - i):
                    distance = position_list[i] - position_list[self.cfg["n_agents"] - 1 - j]
                    distance = sum(abs(distance))
                    if distance < radius:
                        self.total_reward -= 0.2
        # Distance-priority map
        distance_maps = []
        for i, agent in enumerate(self.team):
            distance_map = np.ones(self.map.shape) * -1
            map_plus_coverage = self.map.map + self.map.coverage
            free_indices = np.array(np.where(map_plus_coverage == 0))
            for i in range(len(free_indices[0])):
                y, x = free_indices[..., i]
                distance_map[y][x] = self.solve_maze(agent.position, (y,x)) / 38.0
                # distance_map[y][x] = sum(abs([y, x] - agent.position)) / 38.0
            distance_maps.append(distance_map)
        # pri_distance_maps_0 = ( - distance_maps[1] - distance_maps[2])
        # pri_distance_maps_1 = ( - distance_maps[0] - distance_maps[2])
        # pri_distance_maps_2 = ( - distance_maps[0] - distance_maps[1])
        pri_distance_maps_0 = (distance_maps[0] - distance_maps[1] - distance_maps[2])
        pri_distance_maps_1 = (distance_maps[1] - distance_maps[0] - distance_maps[2])
        pri_distance_maps_2 = (distance_maps[2] - distance_maps[0] - distance_maps[1])
        distance_maps = np.stack([pri_distance_maps_0, pri_distance_maps_1, pri_distance_maps_2])
        # Agent-density map
        density_map = np.zeros(self.map.shape, dtype=np.float32)
        for i, agent in enumerate(self.team):
            ay, ax = agent.position
            for j in range(max(0, ay-radius), min(self.map.height, ay+radius)):
                for k in range(max(0, ax-radius), min(self.map.width, ax+radius)):
                    if sum(abs([j, k] - agent.position)) < radius:
                        density_map[j][k] += 1
        # Uncovered area density
        uncovered_area = (self.map.coverage + self.map.map) == 0
        uncovered_area = uncovered_area.astype(np.float32)
        area_density_map = ndimage.uniform_filter(uncovered_area, size=5, mode='constant')
        uncovered_density_map = (area_density_map* 2/3) * uncovered_area
        # Distance-UncoveredArea Mixed
        mixed_map = [uncovered_density_map + distance_maps[i] for i in range(len(distance_maps))]
        # Agent-position map
        agent_pos_maps = [np.zeros(self.map.shape, dtype=np.float32) for _ in range(self.cfg["n_agents"])]
        agents_pos_map_g = np.zeros(self.map.shape, dtype=np.float32)
        for i, agent in enumerate(self.team):
            for j in range(len(agent_pos_maps)):
                if j == i:
                    agent_pos_maps[j][agent.position[Y], agent.position[X]] = 2
                else:
                    agent_pos_maps[j][agent.position[Y], agent.position[X]] = 1
            agents_pos_map_g[agent.position[Y], agent.position[X]] = agent.agent_id / 3.0
        # State stack (4)...(centered operation commented)
        agents_states = []
        for i, agent in enumerate(self.team):
            agent_state = np.stack([self.map.map,
                                    self.map.coverage,
                                    agent_pos_maps[i],
                                    density_map,
                                    mixed_map[i]], axis=-1)
            # if self.centering:
            #     state_output_shape = np.array([self.cfg['state_size']] * 2, dtype=int)
            #     agent_state_centered = []
            #     for i, state in enumerate(agent_state):
            #         centered_state = agent.to_coordinate_frame(state, state_output_shape, fill=0)
            #         agent_state_centered.append(centered_state)
            agents_states.append(agent_state)
        # Global state (4)
        global_state = np.stack([self.map.map, self.map.coverage > 0, agents_pos_map_g, density_map], axis=-1)
        # use axis=-1 (because tensor(batch, width, height, channel)
        state = {
            'agent_0': tuple(
                [agents_states[0],
                 global_state,
                 ]),
            'agent_1': tuple(
                [agents_states[1],
                 global_state,
                 ]),
            'agent_2': tuple(
                [agents_states[2],
                 global_state,
                 ]),
        }
        reward = {
            'agent_0': total_rewards_per_step / 3.0,
            'agent_1': total_rewards_per_step / 3.0,
            'agent_2': total_rewards_per_step / 3.0,
        }
        # Termination statement
        world_terminator = self.timestep == self.termination or self.map.get_coverage_fraction() == 1.0
        # all_done = all(dones) or world_terminator
        all_done = world_terminator
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
        return state, reward, done, info


    def solve_maze(self, start, goal):
        ay, ax = start
        agent_pos = (ay, ax)
        foundPath = list(self.mazesolver.astar(agent_pos, goal))
        return len(foundPath)

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
        plt.pause(0.01)
        return self.fig
