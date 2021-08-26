from gym.spaces import Box, Discrete, Tuple
import numpy as np
import copy

from ray.rllib.models.torch.misc import normc_initializer as \
    torch_normc_initializer, SlimFC
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import one_hot
from ray.rllib.models.modelv2 import restore_original_dimensions

torch, nn = try_import_torch()
DEFAULT_OPTIONS = {
    "fcnet_hiddens": [128, 128],
    "fcnet_activation": "tanh",
    "activation": "relu",
    "conv_compression": 32,
    "conv_filters":  [[8, [4, 4], 2], [16, [4, 4], 2], [32, [3, 3], 2]],
    "value_conv_compression": 128,
    "value_conv_filters":  [[8, [4, 4], 2], [16, [4, 4], 2], [32, [3, 3], 2]],
    "conv_activation": "relu"
}
class ComplexInputNetworkandCentrailzedCritic(TorchModelV2, nn.Module):
    """TorchModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).
    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.
    The data flow is as follows:
    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and vaulue heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        self.original_space = obs_space.original_space if \
            hasattr(obs_space, "original_space") else obs_space

        #assert isinstance(self.original_space, (Tuple)), \
            #"`obs_space.original_space` must be Tuple!"

        nn.Module.__init__(self)
        TorchModelV2.__init__(self, self.original_space, action_space,
                              num_outputs, model_config, name)

        self.action_space = action_space
        # Build the CNN(s) given obs_space's image components.
        self.cnns = {}
        self.vf_cnns = {}

        self.one_hot = {}
        self.flatten = {}
        self.n_agents = 3
        concat_size = 0
        vf_concat_size = 0

        for i, component in enumerate(self.original_space):
            # Image space.
            if len(component.shape) == 3:
                if i == 0:
                    config = {
                        "conv_filters": model_config["conv_filters"]
                        if "conv_filters" in model_config else
                        get_filter_config(obs_space.shape),
                        "conv_activation": model_config.get("conv_activation"),
                        "post_fcnet_hiddens": [],
                    }
                    cnn = ModelCatalog.get_model_v2(
                        component,
                        action_space,
                        num_outputs=None,
                        model_config=config,
                        framework="torch",
                        name="cnn_{}".format(i))
                    concat_size += cnn.num_outputs  # channel x final output shape(2x2)
                    self.cnns[i] = cnn
                    self.add_module("cnn_local", cnn)
                elif i == 2:
                    config = {
                        "conv_filters": [[64, [4, 4], 2], [128, [4, 4], 2], [256, [3, 3], 2], [512, [3, 3], 1]],
                        "conv_activation": model_config.get("conv_activation"),
                        "post_fcnet_hiddens": [],
                    }
                    vf_cnn = ModelCatalog.get_model_v2(
                        component,
                        action_space,
                        num_outputs=None,
                        model_config=config,
                        framework="torch",
                        name="cnn_global")
                    vf_concat_size = vf_cnn.num_outputs
                    self.vf_cnns[i] = vf_cnn
                    self.add_module("cnn_global_critic", vf_cnn)
                else:
                    self.cnns[i] = self.cnns[0]
                    concat_size += self.cnns[0].num_outputs

            # Discrete inputs -> One-hot encode.
            elif isinstance(component, Discrete):
                self.one_hot[i] = True
                concat_size += component.n
            # Everything else (1D Box).
            else:
                self.flatten[i] = int(np.product(component.shape))
                concat_size += self.flatten[i]

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", [128, 128]),
            "fcnet_activation": model_config.get("post_fcnet_activation",
                                                 "relu")
        }
        self.post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"),
                float("inf"),
                shape=(concat_size, ),
                dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack")

        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None

        if num_outputs:
            # Action-distribution head.
            self.logits_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs,
                out_size=num_outputs,
                activation_fn=None,
            )
            # Create the value branch model.
            self.value_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs,
                out_size=1,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01))
        else:
            self.num_outputs = concat_size

        # Centralized critic post fc layer
        # one-shot (Discrete action space)
        vf_concat_size += 5 * self.n_agents

        self.vf_post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"),
                float("inf"),
                shape=(vf_concat_size,),
                dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="vf_post_fc_stack")

        self.central_vf_layer = SlimFC(
                in_size=self.vf_post_fc_stack.num_outputs,
                out_size=1,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01))

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Push image observations through our CNNs.
        outs = []
        for i, component in enumerate(input_dict["obs"]):
            if i in self.cnns:   # use in because self.cnns is a dict
                cnn_out, _ = self.cnns[i]({"obs": component})
                outs.append(cnn_out)
            elif i in self.flatten:
                outs.append(torch.reshape(component, [-1, self.flatten[i]]))

        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack({"obs": out}, [], None)

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        # Logits- and value branches.
        logits, values = self.logits_layer(out), self.value_layer(out)
        self._value_out = torch.reshape(values, [-1])
        return logits, []

    def central_value_function(self, obs, action, other_actions):
        original_obs = restore_original_dimensions(obs, self.original_space, "torch")
        outs = []

        vf_cnn_out, _ = self.vf_cnns[2]({"obs": original_obs[2]})
        outs.append(vf_cnn_out)
        flatten_self_acts = one_hot(action, self.action_space)
        outs.append(flatten_self_acts)
        flatten_other_acts = one_hot(other_actions, self.action_space).reshape(original_obs[0].size()[0], (self.n_agents-1) * 5)
        outs.append(flatten_other_acts)

        out = torch.cat(outs, dim=1)
        out, _ = self.vf_post_fc_stack({"obs": out}, [], None)
        central_value = self.central_vf_layer(out)
        return torch.reshape(central_value, [-1])

    @override(ModelV2)
    def value_function(self):
        return self._value_out
