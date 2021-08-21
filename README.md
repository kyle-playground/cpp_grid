# ccp_grid [use rllib]
Coverage Path Planning based on reinforcement learning for three explorer in unknown obstacle grid world.

Agents share the same policy and their observations are concatenated for each agent assuming communication exists.

PPO and centralized critic are adopted for the policy, the setting is overall a multi-agent task with independent learners. 
