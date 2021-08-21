# Experiment result documentation

### Hyperparameter Searching
**Experiment1**: Setting1, rollout length = 32, **R**: reward decreasing

**Experiment2**: Setting1, rollout length = 100, **R**: reward increased but drop afterward


### Different observation setting

1. <br>**Observation** = [self local map, self coverage, others' local map, others' coverage, self position, others' position]<br>
**Global State** = [Map, coverage from all agents, agents' position map]
2. <br>**Observation** = [Map, self coverage, others' coverage, self position, others' position]<br>
**Global State** = [Map, coverage from all agents, agents' position map]