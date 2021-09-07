# To test if code work
import pylab as pl

from gridworld import CoverageEnv
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
env = CoverageEnv()
env.reset()

for i in range(100):
    act_dict = {
        "agent_0": np.random.randint(1, 5),
        "agent_1": np.random.randint(1, 5),
        "agent_2": np.random.randint(1, 5),
    }
    obs,_,_,_ = env.step(act_dict)
#
#
# # green:0 blue:1 purple:2
hsv = np.ones((3, 3))
hsv[..., 0] = np.linspace(160 / 360, 250 / 360, 3)
team_agents_color = colors.hsv_to_rgb(hsv)
all_team_colors = [(0, 0, 0, 0)] + [tuple(list(c) + [0.5]) for c in team_agents_color]

map_colormap = colors.ListedColormap(['white', 'black', 'grey'])

local_map = map_colormap(obs["agent_0"][0][..., 0])
coverable = map_colormap(obs["agent_0"][0][..., 1])
agent_pos_map = obs["agent_0"][0][..., 2]


map = map_colormap(obs["agent_0"][1][..., 0])
coverage = obs["agent_0"][1][..., 1]
agent_p_map = colors.ListedColormap(all_team_colors)(obs["agent_0"][1][..., 2])


plt.subplot(2,3,1)
plt.imshow(map)
plt.subplot(2,3,2)
plt.imshow(coverage)
plt.subplot(2,3,3)
plt.imshow(agent_p_map)

plt.subplot(2,3,4)
plt.imshow(local_map)
plt.subplot(2,3,5)
plt.imshow(coverable)
plt.subplot(2,3,6)
plt.imshow(agent_pos_map)
plt.show()





