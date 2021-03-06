"""An example of customizing PPO to leverage a centralized critic.
Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.
Compared to simply running `rllib/examples/two_step_game.py --run=PPO`,
this centralized critic version reaches vf_explained_variance=1.0 more stably
since it takes into account the opponent actions as well as the policy's.
Note that this is also using two independent policies instead of weight-sharing
with one.
See also: centralized_critic_2.py for a simpler approach that instead
modifies the environment.
"""

import argparse
import random
import numpy as np
from datetime import datetime
import os
import yaml
import platform
import pandas as pd

import ray
from ray import tune
from ray.tune import sample_from
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.tune.schedulers.pb2 import PB2

from gridworld import CoverageEnv
from model import ComplexInputNetworkandCentrailzedCritic


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

OBS = "obs"
OTHER_AGENTS_ACTION = "other_agents_actions"
SELF_ID = "self_id"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")

parser.add_argument("--max", type=int, default=10000000)
parser.add_argument("--num_samples", type=int, default=4)
parser.add_argument("--perturb", type=float, default=0.25)
parser.add_argument("--t_ready", type=int, default=50000)
parser.add_argument(
        "--criteria", type=str,
        default="timesteps_total")


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function)
        else:
            self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    pytorch = policy.config["framework"] == "torch"

    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        # get others sample batch from other_agent_batches
        agent_id = sample_batch["agent_index"][0]
        agents = ["agent_0", "agent_1", "agent_2"]
        id_map = {"agent_0": 1, "agent_1": 2, "agent_2": 3}
        agents.pop(agent_id)
        other_acts_batch = np.stack(
            [np.stack((other_agent_batches[other_agent_id][1]["actions"],
                      np.ones_like(other_agent_batches[other_agent_id][1]["actions"]) * id_map[other_agent_id]), axis=1)
             for other_agent_id in agents], axis=-1)

        # record all actions in the trajectory
        sample_batch[OTHER_AGENTS_ACTION] = other_acts_batch
        sample_batch[SELF_ID] = np.ones_like(sample_batch["agent_index"]) * (agent_id+1).astype(np.float32)

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(
                sample_batch[OBS], policy.device),
            convert_to_torch_tensor(
                sample_batch[SampleBatch.ACTIONS], policy.device),
            convert_to_torch_tensor(
                sample_batch[SELF_ID], policy.device),
            convert_to_torch_tensor(
                sample_batch[OTHER_AGENTS_ACTION], policy.device)) \
            .cpu().detach().numpy()

    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OTHER_AGENTS_ACTION] = np.zeros_like(
            np.stack((np.c_[sample_batch[SampleBatch.ACTIONS], sample_batch[SampleBatch.AGENT_INDEX]],
                      np.c_[sample_batch[SampleBatch.ACTIONS], sample_batch[SampleBatch.AGENT_INDEX]],), axis=-1))
        sample_batch[SELF_ID] = np.zeros_like(sample_batch[SampleBatch.AGENT_INDEX])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])

    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = torch_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SELF_ID],
        train_batch[OTHER_AGENTS_ACTION])

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved
    return loss


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out),
    }


CCPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="CCPPOTorchPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_init=setup_torch_mixins,
    mixins=[
        TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
        CentralizedValueMixin
    ])


def get_policy_class(config):
    if config["framework"] == "torch":
        return CCPPOTorchPolicy

CCTrainer = PPOTrainer.with_updates(
    name="CCPPOTrainer",
    default_policy=CCPPOTorchPolicy,
    get_policy_class=get_policy_class,
)

if __name__ == "__main__":
    mac_test = platform.system() == "Darwin"
    if mac_test:
        ray.init()
    else:
        # For server
        ray.init(num_cpus=40, num_gpus=4)

    args = parser.parse_args()
    with open('config.yaml', "rb") as config_file:
        coverage_config = yaml.load(config_file, Loader=yaml.FullLoader)

    if mac_test:
        test_config = {
            "rollout_fragment_length": 8,
            "train_batch_size": 64,
            "sgd_minibatch_size": 8,
            "num_sgd_iter": 8,
            "num_workers": 1,
            "num_gpus": 0,
            "num_gpus_per_worker": 0,
        }
        coverage_config.update(test_config)

    # Register custom model and environment
    register_env("coverage", lambda config: CoverageEnv(config))
    ModelCatalog.register_custom_model("cc_model", ComplexInputNetworkandCentrailzedCritic)

    config = {
        "multiagent": {
            "policies": {
                "shared_policy": (None, CoverageEnv.single_agent_merge_obs_space,
                                  CoverageEnv.single_agent_action_space,
                                  {"framework": "torch"}),
            },
            "policy_mapping_fn": (lambda aid: "shared_policy"),
            "count_steps_by": "env_steps",
        },
    }
    config.update(coverage_config)

    pb2 = PB2(
        time_attr=args.criteria,
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=args.t_ready,
        quantile_fraction=args.perturb,  # copy bottom % with top %
    #     Specifies the hyperparam search space
        hyperparam_bounds={
            "lambda": [0.9, 1.0],
            "clip_param": [0.1, 0.5],
            "lr": [5e-5, 1e-7],
        })
    tune_config = {
        "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
        "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.3)),
        "lr": sample_from(lambda spec: random.uniform(5e-5, 1e-7)),
    }
    config.update(tune_config)

    args.dir = "cpp_pb2_Size{}".format(str(args.num_samples))

    timelog = str(datetime.date(datetime.now())) + "_" + str(
        datetime.time(datetime.now()))

    analysis = tune.run(CCTrainer,
                        name="cpp_pb2_{}".format(timelog),
                        scheduler=pb2,
                        num_samples=args.num_samples,
                        config=config,
                        stop={args.criteria: args.max},
                        verbose=1,
                        local_dir="./log",
                        checkpoint_at_end=True,   # add check point to save model
                        )

    all_dfs = analysis.trial_dataframes
    names = list(all_dfs.keys())

    results = pd.DataFrame()
    for i in range(args.num_samples):
        df = all_dfs[names[i]]
        df = df[[
            "timesteps_total", "episodes_total", "episode_reward_mean",
        ]]
        results = pd.concat([results, df]).reset_index(drop=True)

    if args.save_csv:
        if not (os.path.exists("log/" + args.dir)):
            os.makedirs("log/" + args.dir)
        results.to_csv("log/{}/cpp_pb2.csv".format(args.dir))
