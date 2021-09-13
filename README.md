# cpp_grid (Original intention) [use rllib] (2021.09.09)
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
Map, Coverable Area, Agent Position Map: (All centered)     
</pre>
<img width="250" alt="Screen Shot 2021-09-09 at 11 29 46 AM" src="https://user-images.githubusercontent.com/64893909/132617931-683e64cd-63a5-479d-8237-b274ea8e352f.png"><img width="250" alt="Screen Shot 2021-09-09 at 11 29 55 AM" src="https://user-images.githubusercontent.com/64893909/132617956-164f1a8a-38e2-4a50-afc5-215735a88219.png"><img width="250" alt="Screen Shot 2021-09-09 at 11 30 02 AM" src="https://user-images.githubusercontent.com/64893909/132617962-11f50c5d-2f5b-4905-9c8b-d8d3db8d0684.png">

## Observation(state) for centralized critic:
<pre>
Map, Covered Density Map (lighter place being covered more), Agent Position Map
+ Actions from all agents:
</pre>
<img width="250" alt="Screen Shot 2021-09-09 at 11 29 28 AM" src="https://user-images.githubusercontent.com/64893909/132618071-8060f88a-ad6c-4f2a-a14e-959aa2b3c55c.png"><img width="250" alt="Screen Shot 2021-09-09 at 11 29 33 AM" src="https://user-images.githubusercontent.com/64893909/132618074-121eaeee-5cf2-440d-8a79-51dddf232b5b.png"><img width="250" alt="Screen Shot 2021-09-09 at 11 29 40 AM" src="https://user-images.githubusercontent.com/64893909/132618080-1fe7f18a-df0e-4c73-b3b4-c0c07ef68417.png">

## Result

![Untitled](https://user-images.githubusercontent.com/64893909/132844660-1924d69f-021a-4e59-bc3c-6e733a348ad0.gif)

max reward: 115
Total_steps_per_round: 150

![Screenshot 2021-09-10 17_44_26](https://user-images.githubusercontent.com/64893909/132844721-91ad5c95-225e-41a7-b7d0-fb75ff91fb30.png)
