from gridworld import CoverageEnv
import matplotlib.pyplot as plt


env = CoverageEnv()
env.reset()
for i in range(3):
    act_dict = {
        "agent_0": 1,
        "agent_1": 2,
        "agent_2": 3,
    }
    obs,_,_,_ = env.step(act_dict)
obs_print = obs["agent_0"][0].reshape(6, 24, 24)
print(obs["agent_0"][0][..., 0])
plt.imshow(obs["agent_0"][0][..., 0])
plt.show()
