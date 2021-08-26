# To test if code work
from gridworld import CoverageEnv
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
env = CoverageEnv()
env.reset()
for i in range(150):
    act_dict = {
        "agent_0": np.random.randint(1, 5),
        "agent_1": np.random.randint(1, 5),
        "agent_2": np.random.randint(1, 5),
    }
    obs,_,_,_ = env.step(act_dict)
hsv = np.ones((3, 3))
hsv[..., 0] = np.linspace(160 / 360, 250 / 360, 3)
team_agents_color = colors.hsv_to_rgb(hsv)
all_team_colors = [(0, 0, 0, 0)] + [tuple(list(c) + [0.5]) for c in team_agents_color]

map_colormap = colors.ListedColormap(['white', 'black', 'gray'])

local_map = map_colormap(obs["agent_0"][0][..., 0])
local_coverage = map_colormap(obs["agent_0"][0][..., 1])

map = map_colormap(env.map.map)
coverage = colors.ListedColormap(all_team_colors)(env.map.coverage)
agent_map = colors.ListedColormap(all_team_colors)(obs["agent_0"][2][..., 2])

# plt.imshow(map)
# plt.show()
# plt.imshow(coverage)
# plt.show()
# plt.imshow(agent_map)
# plt.show()
plt.imshow(local_map)
plt.show()
plt.imshow(local_coverage)
plt.show()
