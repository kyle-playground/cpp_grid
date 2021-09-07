# cpp_grid (MA Heuristic setting) [use rllib] (2021.09.07)
Environment code modified from https://github.com/proroklab/adversarial_comms

Coverage Path Planning based on reinforcement learning for three explorers in unknown (or known) obstacle grid world.

Agents share the same policy and infomation can be shared between agents.

PPO and centralized critic are adopted for the policy, the setting is overall a multi-agent task with an independent learner. 

## Reward setting
<pre>
Reward = 1      If new grid, -0.2 if revisited

</pre>
## Observation(state) for policy:
#### Map remains known for right now (experimenting phase) 
<pre>
Map, Merged Coverage, Agent Density Map(radius=3), Distance-Border Cost Mixed Map, Position Map:             
</pre>



## Observation(state) for centralized critic:
<pre>
Map, Global Coverage, Agent Position Map(for revisit penalty), Agent Density Map(for redundancy penalty) 

+ Actions from all agents:
</pre>


## Result

