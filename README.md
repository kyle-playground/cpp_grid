# cpp_grid (MA setting) [use rllib] (2021.08.23)
Environment code modified from https://github.com/proroklab/adversarial_comms

Coverage Path Planning based on reinforcement learning for three explorers in unknown (or known) obstacle grid world.

Agents share the same policy and infomation can be shared between agents.

PPO and centralized critic are adopted for the policy, the setting is overall a multi-agent task with an independent learner. 

## Reward setting
<pre>
Reward = 1      If new grid, -0.2 if revisited

Reward += -0.1  If two agents are too close (Intersect appear in density map)

Reward += 0.1   If agent move toward the center of darker area
</pre>
## Observation(state) for policy:
#### Map remains known for right now (experimenting phase) 
<pre>
Map, Merged Coverage, Agent Density Map(radius=3), Distance-Border Cost Mixed Map, Position Map:             
</pre>
<img width="250" alt="Screen Shot 2021-09-07 at 11 43 13 AM" src="https://user-images.githubusercontent.com/64893909/132281287-223a305f-921a-4586-b4f3-353e473346b8.png"><img width="250" alt="Screen Shot 2021-09-07 at 11 43 20 AM" src="https://user-images.githubusercontent.com/64893909/132281406-8b0ff1ee-a3dc-42c8-830e-3b7ab6888084.png"><img width="250" alt="Screen Shot 2021-09-07 at 11 43 34 AM" src="https://user-images.githubusercontent.com/64893909/132281413-9e6f7625-5804-4e1e-a961-4246083d79cd.png"><img width="250" alt="Screen Shot 2021-09-07 at 11 43 41 AM" src="https://user-images.githubusercontent.com/64893909/132281441-e2f2d9ff-4499-49e8-a62b-8db7c17ebfd3.png"><img width="250" alt="Screen Shot 2021-09-07 at 3 55 34 PM" src="https://user-images.githubusercontent.com/64893909/132306729-241ab203-f587-4cbb-b7a1-2e847f96492f.png">


## Observation(state) for centralized critic:
<pre>
Map, Global Coverage, Agent Position Map(for revisit penalty), Agent Density Map(for redundancy penalty) 

+ Actions from all agents:
</pre>
<img width="250" alt="Screen Shot 2021-09-07 at 11 43 13 AM" src="https://user-images.githubusercontent.com/64893909/132281556-00e19d3e-b6fd-4a8d-9b91-e75285eab94f.png"><img width="250" alt="Screen Shot 2021-09-07 at 11 43 20 AM" src="https://user-images.githubusercontent.com/64893909/132281559-99eedaf6-e707-4564-84db-a8ff374429d2.png"><img width="250" alt="Screen Shot 2021-09-07 at 11 43 27 AM" src="https://user-images.githubusercontent.com/64893909/132281563-de11b3b5-7ad6-4970-9e73-b590638d868b.png"><img width="250" alt="Screen Shot 2021-09-07 at 11 43 34 AM" src="https://user-images.githubusercontent.com/64893909/132281570-a54293e7-b171-4542-afc9-9d5cd89930fb.png">

## Distance-Border Cost Mixed Map 
Hope agent explore darker area first

Pic1. DBCMM of agent1 (green)

Pic2. DBCMM of agent2 (blue)

Pic3. DBCMM of agent3 (purple)

<img width="250" alt="Screen Shot 2021-09-07 at 11 43 41 AM" src="https://user-images.githubusercontent.com/64893909/132281671-ee0f4a59-2f86-41b7-a0a1-b90c147dd91b.png"><img width="250" alt="Screen Shot 2021-09-07 at 11 43 46 AM" src="https://user-images.githubusercontent.com/64893909/132281701-d43f33c5-663b-4dc2-8810-9accd4089cd2.png"><img width="250" alt="Screen Shot 2021-09-07 at 11 43 53 AM" src="https://user-images.githubusercontent.com/64893909/132281727-56edb0d2-04c9-408e-a619-398322d753f0.png">

## Result

