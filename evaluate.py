import collections
import json
from pathlib import Path
import traceback

import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.registry import register_env

from gridworld import CoverageEnv
from model import ComplexInputNetworkandCentrailzedCritic
"""
best_checkpoint = results.get_best_checkpoint(
    results.trials[0], mode="max")
print(f".. best checkpoint was: {best_checkpoint}")
"""
def update_dict(d, u):
    for k, v in u.items():
        #?????????????
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def initialize():
    ray.init()
    register_env("coverage", lambda config: CoverageEnv(config))
    ModelCatalog.register_custom_model("cc_model", ComplexInputNetworkandCentrailzedCritic)


def run_trial(trainer_class=PPOTrainer, checkpoint_path=None, cfg_update={}, render=False):
    try:
        if checkpoint_path is not None:
            with open(Path(checkpoint_path).parent / "params.json") as json_file:
                cfg = json.load(json_file)

        cfg = update_dict(cfg, cfg_update)

        trainer = trainer_class(
            env=cfg['env'],
            config={
                "framework": "torch",
                "num_workers": 0,
                "env_config": cfg['env_config'],
                "model": cfg['model']
            }
        )
        env = CoverageEnv(cfg['env_config'])
        obs = env.reset()
        for i in range(env.termination):
            action_dict = {}
            for agent_id, obs in obs.items():
                action_dict[agent_id] = trainer.compute_action(obs)
            obs, reward, done, info = env.step(action_dict)
            if render:
                env.render()
        print("agent_0")
        print(info['agent_0'])
        print("agent_1")
        print(info['agent_1'])
        print("agent_2")
        print(info['agent_2'])

    except Exception as e:
        print(e, traceback.format_exc())
        raise

if __name__ == "__main__":
    checkpoint_path = "log/CCPPOTrainer_2021-08-19_13-06-48/CCPPOTrainer_coverage_41f26_00000_0_2021-08-19_13-06-48/checkpoint_000251"
    initialize()
    run_trial(checkpoint_path=checkpoint_path)


