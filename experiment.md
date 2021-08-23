# Experiment result documentation

### Hyperparameter Searching
**Experiment1**: Setting1, rollout length = 32, **R**: reward decreasing

**Experiment2**: Setting1, rollout length = 128, **R**: reward increased but drop afterward


### Different observation setting

1. <br>**Observation** = [Conv(self local map, self coverage, others' local map, others' coverage), self position, others' positions]<br>
**Global State** = [Conv(Map, global coverage map, agents' position map)]
2. <br>**Observation** = [Conv(Map, self coverage, others' coverage), self position, others' positions]<br>
**Global State** = [Conv(Map, global coverage map, agents' position map)]
3. <br>**Rward** = -0.2, If freeze or revisist same area
4. <br>**Observation** = [Conv(map_i,coverage_i) for i in team, self position, others' positions] <br> local map is not marked     <br>**Global State** = [Conv(Map, global coverage map, agents' position map)]