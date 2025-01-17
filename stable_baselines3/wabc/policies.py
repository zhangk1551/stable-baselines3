from typing import Any, Dict, List, Optional, Type

import torch as th
import numpy as np
from gymnasium import spaces
from torch import nn
from scipy.ndimage import zoom

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    EgoCentricCNN,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.utils import get_grid_action_prob, check_static

from torchdriveenv.diffusion_expert import DiffusionExpert


class WABCPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    b_net: ContinuousCritic
    b_net_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = EgoCentricCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.diffusion_expert = DiffusionExpert("pretrained_edm_module/model.ckpt")

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.b_net = self.make_b_net()
        self.b_net_target = self.make_b_net()
        self.b_net_target.load_state_dict(self.b_net.state_dict())
        self.b_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.b_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_b_net(self) -> ContinuousCritic:
        # Make sure we always have separate networks for features extractors etc
        print("enter make b net")
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
#        net_args = self._update_features_extractor(self.net_args)
        print(net_args)
        return ContinuousCritic(**net_args).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        action, prob = self.sample(obs, deterministic=deterministic)
#        print("action in _predict")
#        print(action)
        return action.squeeze(dim=1)

    def get_w_grid(self, obs: PyTorchObs, nticks: int = 10, interpolation_num: int = 10,
                   vmin_x = -1.0, vmax_x = 1.0, vmin_y = -1.0, vmax_y = 1.0) -> th.Tensor:
        with th.no_grad():
            if obs.max() > 1:
                obs = obs.float() / 255.0
            b_grid = get_grid_action_prob(fn=self.b_net_target.q1_forward, observation=obs,
                                          vmin_x=vmin_x, vmax_x=vmax_x, vmin_y=vmin_y, vmax_y=vmax_y,
                                          nticks=nticks, interpolation_num=interpolation_num)
    #        print("b_grid")
    #        print(b_grid.shape)
            c_grid = get_grid_action_prob(fn=self.diffusion_expert.module.diffusion.p_energy, observation=obs,
                                          vmin_x=vmin_x, vmax_x=vmax_x, vmin_y=vmin_y, vmax_y=vmax_y,
                                          nticks=nticks, interpolation_num=interpolation_num, p_energy=True)
    #        print("c_grid")
    #        print(c_grid.shape)
            w_grid = b_grid * c_grid

            w_grid = w_grid / w_grid.sum(dim=(1, 2), keepdim=True)
        return w_grid


    def get_b_grid(self, obs: PyTorchObs, nticks: int = 10, interpolation_num: int = 10,
                   vmin_x = -1.0, vmax_x = 1.0, vmin_y = -1.0, vmax_y = 1.0) -> th.Tensor:
        with th.no_grad():
            if obs.max() > 1:
                obs = obs.float() / 255.0
            b_grid = get_grid_action_prob(fn=self.b_net_target.q1_forward, observation=obs,
                                          vmin_x=vmin_x, vmax_x=vmax_x, vmin_y=vmin_y, vmax_y=vmax_y,
                                          nticks=nticks, interpolation_num=interpolation_num)
        return b_grid


    def get_c_grid(self, obs: PyTorchObs, nticks: int = 10, interpolation_num: int = 10,
                   vmin_x = -1.0, vmax_x = 1.0, vmin_y = -1.0, vmax_y = 1.0) -> th.Tensor:
        with th.no_grad():
#            print("obs in get_c_grid")
#            print(obs)
            if obs.max() > 1:
                obs = obs.float() / 255.0
            c_grid = get_grid_action_prob(fn=self.diffusion_expert.module.diffusion.p_energy, observation=obs,
                                          vmin_x=vmin_x, vmax_x=vmax_x, vmin_y=vmin_y, vmax_y=vmax_y,
                                          nticks=nticks, interpolation_num=interpolation_num, p_energy=True)
        return c_grid


    def get_c_grid_from_samples(self, obs: PyTorchObs, nticks: int = 10, interpolation_num: int = 10,
                   vmin_x = -1.0, vmax_x = 1.0, vmin_y = -1.0, vmax_y = 1.0) -> th.Tensor:
        with th.no_grad():
            print("obs shape")
            print(obs)
            batch_size = obs.shape[0]
#            samples = self.diffusion_expert.module.diffusion.sample(s=obs, n=10000, shape=[2], is_batch=True)
            samples = self.diffusion_expert.module.diffusion.sample(s=obs.float().squeeze(), n=10000, shape=[2])
            p = np.histogram2d(samples[:, 1], samples[:, 0], bins=nticks, range=[[vmin_x, vmax_x], [vmin_y, vmax_y]])
            print("c_grid_from_samples: p")
            print(p)
            p = p / np.sum(p, axis=(1, 2), keepdims=True)
            p = p * 0.8 + 0.2
            interpolated_p = th.Tensor(np.stack([zoom(p[i], (interpolation_num, interpolation_num), order=1) for i in range(batch_size)])).to(obs.device)
        return interpolated_p


    def sample(self, obs: PyTorchObs, sample_num: int = 1, deterministic: bool = False) -> th.Tensor:
        batch_size = obs.shape[0]

        nticks = 11
        interpolation_num = 10
        vmin_x = -1.0
        vmax_x = 1.0
        vmin_y = -1.0
        vmax_y = 1.0
#        deterministic = True

#        probabilities = (self.get_c_grid(obs, nticks, interpolation_num, vmin_x, vmax_x, vmin_y, vmax_y)).view(batch_size, -1)
        probabilities = (self.get_b_grid(obs, nticks, interpolation_num, vmin_x, vmax_x, vmin_y, vmax_y)).view(batch_size, -1)
        probabilities /= th.sum(probabilities, dim=1, keepdims=True)
#        topk_values, topk_indices = th.topk(probabilities, k=5, dim=1)
#        print("topk_indices")
#        print(topk_indices)
#        print("topk_values")
#        print(topk_values)
#        b_grid = (self.get_b_grid(obs, nticks, interpolation_num, vmin_x, vmax_x, vmin_y, vmax_y)).view(batch_size, -1)
#        c_grid = (self.get_c_grid(obs, nticks, interpolation_num, vmin_x, vmax_x, vmin_y, vmax_y)).view(batch_size, -1)
#        probabilities = th.where(check_static(obs).unsqueeze(-1).expand(*b_grid.shape), b_grid, c_grid)

        if deterministic:
            argmax_indices = th.argmax(probabilities, dim=1)
            sampled_indices = argmax_indices.unsqueeze(1).repeat(1, sample_num)
        else:
            sampled_indices = th.multinomial(probabilities, num_samples=sample_num, replacement=True)

        grid_len = nticks * interpolation_num
        acceleration = vmin_y + (sampled_indices // grid_len) / (grid_len - 1) * (vmax_y - vmin_y - (vmax_y - vmin_y) / nticks) + (vmax_y - vmin_y) / nticks * 0.5
        steering = vmin_x + (sampled_indices % grid_len) / (grid_len - 1) * (vmax_x - vmin_x - (vmax_x - vmin_x) / nticks) + (vmax_x - vmin_x) / nticks * 0.5
#        steering = vmin_x + ((sampled_indices % grid_len) / grid_len * (vmax_x - vmin_x)) + (vmax_x - vmin_x) / nticks * 0.5
#            actions = th.tensor([acceleration, steering], device=self.device).reshape(batch_size, sample_num, 2)
        actions = th.cat([acceleration.unsqueeze(-1), steering.unsqueeze(-1)], dim=-1) #.reshape(batch_size, sample_num, 2)
#        print("steering")
#        print(steering)
#        print("acceleration")
#        print(acceleration)
#        print("actions")
#        print(actions)

        action_prob = th.gather(probabilities, dim=1, index=sampled_indices) # Shape: [batch_size, sample_num]
#            print("action_prob")
#            print(action_prob.shape)
#        default_actions = th.rand((batch_size, sample_num)).unsqueeze(-1).expand((batch_size, sample_num, 2)).to(self.device) * 2 - 1.0
#        default_action_prob = th.tensor([1.0], device=self.device).expand((batch_size, sample_num)).to(self.device)
#
#        dice = th.rand(batch_size, sample_num).to(self.device)
#        action_prob = th.where(dice < 0.05, default_action_prob, action_prob)
#        dice = dice.unsqueeze(-1).expand((batch_size, sample_num, 2))
#        actions = th.where(dice < 0.05, default_actions, actions)
#
#        actions[..., 1] *= 0.3
#        noise = (th.rand_like(actions) - 0.5) / nticks
#        actions += noise
#        print("actions")
#        print(actions)
#        print("action_prob")
#        print(action_prob)

        return actions, action_prob

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.b_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = WABCPolicy


class CnnPolicy(WABCPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = EgoCentricCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MultiInputPolicy(WABCPolicy):
    """
    Policy class for DQN when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
