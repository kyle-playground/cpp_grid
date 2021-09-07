# cpp_grid (MA setting) [use rllib] (2021.08.23)
Environment code modified from https://github.com/proroklab/adversarial_comms

Coverage Path Planning based on reinforcement learning for three explorers in unknown obstacle grid world.

Agents share the same policy and their observations(conv compressed) are concatenated for action inference assuming communication exists.

PPO and centralized critic are adopted for the policy, the setting is overall a multi-agent task with an independent learner. 

Reward = 1 if new grid, -0.2 if revisited

## Observation(state) for policy:



## Observation(state) for centralized critic:
Global map, global coverage, and agent position map + Actions from all agents:


## Result

