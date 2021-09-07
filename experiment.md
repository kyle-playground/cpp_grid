# Experiment result documentation

### Hyperparameter Searching



### Different observation setting

1. <br>**Observation** = [Conv(self local map, self coverage, others' local map, others' coverage), self position, others' positions]<br>
**Global State** = [Conv(Map, global coverage map, agents' position map)]
2. <br>**Observation** = [Conv(Map, self coverage, others' coverage), self position, others' positions]<br>
**Global State** = [Conv(Map, global coverage map, agents' position map)]
3. <br>**Observation** = [Conv(map_i,coverage_i) for i in team, self position, others' positions] <br> local map is not marked     <br>**Global State** = [Conv(Map, global coverage map, agents' position map)]
4. <br>**Observation** = [Conv(Merged map, Merged coverage), self position, others' positions]

### Different reward setting

1. <br> reward = 1, new grid
<br> reward = -0.2, revisit