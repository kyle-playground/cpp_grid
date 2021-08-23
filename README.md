# cpp_grid [use rllib] task fail (2021.08.23)
Coverage Path Planning based on reinforcement learning for three explorer in unknown obstacle grid world.

Agents share the same policy and their observations are concatenated for each agent assuming communication exists.

PPO and centralized critic are adopted for the policy, the setting is overall a multi-agent task with independent learners. 

Reward = 1 if new grid, -0.2 if revisited

## Observation(state) for policy:
Local map and local coverage (2) from all agents (3), and position (X, Y, 2 if self else 1):

<img width="400" alt="Screen Shot 2021-08-23 at 2 25 56 PM" src="https://user-images.githubusercontent.com/64893909/130400936-a7e4819b-dfcc-4d32-9ad2-ae6e6fd3ef5e.png"><img width="400" alt="Screen Shot 2021-08-23 at 2 25 51 PM" src="https://user-images.githubusercontent.com/64893909/130400923-58b0d2b5-a7d0-4cef-82ce-db40bfe804fa.png">

## Observation(state) for centralized critic:
Global map, global coverage, and agent position map:

<img width="250" alt="Screen Shot 2021-08-23 at 2 26 19 PM" src="https://user-images.githubusercontent.com/64893909/130401129-75278716-f438-46b9-bc34-38e8da7e90f9.png"><img width="250" alt="Screen Shot 2021-08-23 at 2 26 14 PM" src="https://user-images.githubusercontent.com/64893909/130401147-0a883abc-9206-4a04-8ea4-174f9b89a1a0.png"><img width="250" alt="Screen Shot 2021-08-23 at 2 26 03 PM" src="https://user-images.githubusercontent.com/64893909/130401151-6ad51fe3-512f-4359-8c17-d5b15447e6cc.png">



