from typing import Any, Dict, List, Optional, Type, Union

import gym
import torch
import torch as th
import numpy as np
from torch import nn


from stable_baselines3.common.policies import BasePolicy
# from stable_baselines3.common.torch_layers import (
#     BaseFeaturesExtractor,
#     CombinedExtractor,
#     FlattenExtractor,
#     NatureCNN,
#     create_mlp,
# )
from tools.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    LSHSelfAttention,
    LSHAttention,
    ScaleDotAttention,
    SoftAttention,
    Autopadder,
    LSHSA_Autopadder,
    MlpExtractor,
    NatureCNN,
    create_mlp,
    RNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import obs_as_tensor, get_device


class QNetwork(nn.Module):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        selected_num: int = 10,
        strategy: str = "concat",
        features_dim: int = 64,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        # super().__init__(
        #     observation_space,
        #     action_space,
        #     features_extractor=features_extractor,
        #     normalize_images=normalize_images,
        # )
        super(QNetwork, self).__init__()

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = None
        self.selected_num = selected_num
        self.strategy = strategy
        self.features_dim = features_dim
        self.action_space = action_space
        self.observation_space = observation_space
        self.normalize_images = normalize_images
        action_dim = self.action_space.n  # number of actions
        self._build_network(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        # q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        # self.q_net = nn.Sequential(*q_net)

    def _build_network(self, features_dim, action_dim, net_arch, activation_fn):
        if self.strategy == "concat":
            print("execute the concat strategy")
            self.features_extractor = nn.Flatten()
            self.RNN =RNN(input_shape=1, hidden_dim=features_dim)
            q_net = create_mlp(features_dim*self.observation_space.shape[1], action_dim, net_arch, activation_fn)
            self.q_net = nn.Sequential(*q_net)
        if self.strategy == "solo":
            print("execute the solo strategy")
            self.stock_RNN = RNN(input_shape=1, hidden_dim=features_dim)
            self.index_RNN = RNN(input_shape=1, hidden_dim=features_dim)
            stock_network = create_mlp(features_dim, int(features_dim), net_arch, activation_fn)
            index_network = create_mlp(features_dim, int(features_dim), net_arch, activation_fn)
            self.stock_network = nn.Sequential(*stock_network)
            self.index_network = nn.Sequential(*index_network)

            # fuse_network = create_mlp(32*self.observation_space.shape[1], action_dim, net_arch, activation_fn)            
            # self.fuse_network = nn.Sequential(*fuse_network)
            # self.features_extractor = nn.Flatten()
            
            fuse_network = create_mlp(features_dim*action_dim, action_dim, net_arch, activation_fn)
            score_network = create_mlp(int(features_dim), action_dim, net_arch, activation_fn)
            self.fuse_network = nn.Sequential(*fuse_network)
            self.score_network = nn.Sequential(*score_network)
            self.features_extractor = nn.Flatten()

        if self.strategy == "inter":
            print("execute the interative strategy")
            # self.stock_RNN = RNN(input_shape=1, hidden_dim=features_dim)
            # self.index_RNN = RNN(input_shape=1, hidden_dim=features_dim)
            # stock_network = create_mlp(features_dim, int(features_dim/2), net_arch, activation_fn)
            # index_network = create_mlp(features_dim, int(features_dim/2), net_arch, activation_fn)
            # self.stock_network = nn.Sequential(*stock_network)
            # self.index_network = nn.Sequential(*index_network)
            num_head = 2
            num_layer = 2
            input_dim = self.observation_space.shape[0]
            dim_feedforward = 64
            transformer_args = {
            "num_layers": num_layer,
            "input_dim": self.observation_space.shape[0],
            "dim_feedforward": dim_feedforward,
            "num_heads": num_head,
            "dropout": 0.5,
            }
            # self.encoder = TransformerEncoder(**transformer_args)
            # self.rnn = RNN(input_shape=1, hidden_dim=features_dim)
            self.encoder = nn.TransformerEncoderLayer(d_model=input_dim,nhead=2,batch_first=True,dropout=0.5,dim_feedforward=dim_feedforward)
            self.features_extractor = nn.Flatten()
            fuse_network = create_mlp(input_dim*action_dim, action_dim, net_arch, activation_fn)
            score_network = create_mlp(int(input_dim*1), action_dim, net_arch, activation_fn)
            self.fuse_network = nn.Sequential(*fuse_network)
            self.score_network = nn.Sequential(*score_network)

        if self.strategy == "two":
            input_dim = self.observation_space.shape[0]
            print("execute the two-stream strategy")
            # encoder = LSHSelfAttention(dim = features_dim, heads = 2, bucket_size = 32, n_hashes = 6, causal = False)
            encoder = LSHAttention(bucket_size = 8, n_hashes = 6)
            num_head = 2
            num_layer = 2
            input_dim = self.observation_space.shape[0]
            dim_feedforward = 64
            transformer_args = {
            "num_layers": num_layer,
            "input_dim": self.observation_space.shape[0],
            "dim_feedforward": dim_feedforward,
            "num_heads": num_head,
            "dropout": 0.5,
            }
            self.stock_RNN = RNN(input_shape=1, hidden_dim=features_dim)
            self.index_RNN = RNN(input_shape=1, hidden_dim=features_dim)
            # self.encoder = nn.TransformerEncoderLayer(d_model=input_dim,nhead=2,batch_first=True,dropout=0.5,dim_feedforward=dim_feedforward)
            self.encoder = Autopadder(encoder)
            # self.encoder = LSHSA_Autopadder(encoder)
            self.features_extractor = nn.Flatten()
            # self.scale_dot_att = SoftAttention(input_dim)
            self.scale_dot_att = SoftAttention(features_dim)
            fuse_network = create_mlp(features_dim*action_dim, action_dim, net_arch, activation_fn)
            index_network = create_mlp(int(features_dim*action_dim), action_dim, net_arch, activation_fn)
            self.fuse_network = nn.Sequential(*fuse_network)
            self.index_network = nn.Sequential(*index_network)

    def extract_features(self, obs):
        if self.strategy == "concat":
            batch_size = obs.shape[0]
            stock_num = obs.shape[2]
            lookback = obs.shape[1]
            rnn_hidden = self.RNN(obs.reshape(batch_size*stock_num, lookback))   # (stock_dim, lookback, hidden_dim)
            rnn_hidden = rnn_hidden[:,-1,:].view(batch_size, stock_num, self.features_dim)
            features = self.features_extractor(rnn_hidden)
        if self.strategy == "solo":
            index_data = obs[:,:,0]
            stock_data = obs[:,:,1:]
            lookback = obs.shape[1]
            batch_size = obs.shape[0]
            stock_num = stock_data.shape[2]
            index_rnn_hidden = self.index_RNN(index_data.reshape(batch_size, lookback))
            index_rnn_hidden = index_rnn_hidden[:,-1,:].view(batch_size, self.features_dim)
            stock_rnn_hidden = self.stock_RNN(stock_data.reshape(batch_size*stock_num, lookback))
            stock_rnn_hidden = stock_rnn_hidden[:,-1,:].view(batch_size*stock_num, self.features_dim)
            index_feat = self.index_network(index_rnn_hidden)
            stock_feat = self.stock_network(stock_rnn_hidden)

            # index_feat = index_feat.view(batch_size, int(self.features_dim), 1)
            # stock_feat = stock_feat.view(batch_size, stock_num, int(self.features_dim))
            # features = torch.matmul(stock_feat, index_feat) # features is the q-value
            # features = features.view(batch_size, stock_num)

            stock_feat = stock_feat.view(batch_size, stock_num, int(self.features_dim))
            # index_feat = index_feat.view(batch_size, 1, int(self.features_dim/2))
            flatten_feat = self.features_extractor(stock_feat)
            stock_score = self.fuse_network(flatten_feat)
            index_score = self.score_network(index_feat)
            features = stock_score * index_score

        if self.strategy == "inter":

            # index_feat = index_feat.view(batch_size, 1, int(self.features_dim/2))
            # stock_feat = stock_feat.view(batch_size, stock_num, int(self.features_dim/2))
            # fuse_feat = torch.cat((index_feat,stock_feat),1)
            # features = self.features_extractor(fuse_feat)
            # features = self.fuse_network(features)

            obs = obs.permute(0, 2, 1)
            stock_tran_hidden = self.encoder(obs)
            stock_tran_hidden = stock_tran_hidden[:,1:,:]
            index_feat = stock_tran_hidden[:,0,:]
            flatten_feat = self.features_extractor(stock_tran_hidden)
            stock_score = self.fuse_network(flatten_feat)
            index_score = self.score_network(index_feat)
            features = stock_score * index_score

        if self.strategy == "two":
            obs = obs.permute(0, 2, 1)
            stock_data = obs[:,1:,:]
            index_data = obs[:,0,:]
            lookback = obs.shape[2]
            batch_size = obs.shape[0]
            stock_num = stock_data.shape[1]
            index_rnn_hidden = self.index_RNN(index_data.reshape(batch_size, lookback))
            index_rnn_hidden = index_rnn_hidden[:,-1,:].view(batch_size, self.features_dim)
            stock_rnn_hidden = self.stock_RNN(stock_data.reshape(batch_size*stock_num, lookback))
            stock_rnn_hidden = stock_rnn_hidden[:,-1,:].view(batch_size*stock_num, self.features_dim)
            stock_data = stock_rnn_hidden.view(batch_size, stock_num, self.features_dim)
            index_data = index_rnn_hidden.view(batch_size, self.features_dim)

            stock_reform_hidden = self.encoder(stock_data, stock_data)
            # stock_reform_hidden = self.encoder(stock_data)
            (stock_index_hidden,att) = self.scale_dot_att(index_data,stock_data,stock_data)
            flatten_stock_feat1 = self.features_extractor(stock_reform_hidden) #stocks feat with relation among assets
            flatten_stock_feat2 = self.features_extractor(stock_index_hidden)  #stocks feat relation between index and assets
            stock_score = self.fuse_network(flatten_stock_feat1)
            index_score = self.index_network(flatten_stock_feat2)
            features = 0.9*stock_score + 0.1*index_score

        return features

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        if self.strategy == "concat":
            return self.q_net(self.extract_features(obs))
        if self.strategy == "solo" or self.strategy =="inter" or self.strategy=="two":
            return self.extract_features(obs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        k_actions = q_values.argsort(dim=1, descending=True)[:,:self.selected_num]
        # action = q_values.argmax(dim=1).reshape(-1)
        return k_actions

    # def _get_constructor_parameters(self) -> Dict[str, Any]:
    #     data = super()._get_constructor_parameters()

    #     data.update(
    #         dict(
    #             net_arch=self.net_arch,
    #             features_dim=self.features_dim,
    #             activation_fn=self.activation_fn,
    #             features_extractor=self.features_extractor,
    #         )
    #     )
    #     return data


class DQNPolicy(nn.Module):
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

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        selected_num: int = 10,
        strategy: str="concat",
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[th.device, str] = "auto",
    ):
        # super().__init__(
        #     observation_space,
        #     action_space,
        #     features_extractor_class,
        #     features_extractor_kwargs,
        #     optimizer_class=optimizer_class,
        #     optimizer_kwargs=optimizer_kwargs,
        # )
        super(DQNPolicy, self).__init__()

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.observation_space = observation_space
        self.action_space = action_space
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.selected_num = selected_num
        self.strategy = strategy
        self.device = get_device(device)

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "selected_num": self.selected_num,
            "strategy": self.strategy,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        # self.q_net_target.set_training_mode(False)
        self.q_net_target.train(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        # net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        # return QNetwork(**net_args).to(self.device)
        return QNetwork(**self.net_args).cuda()

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def predict(self, obs: th.Tensor, 
        state=None, 
        episode_start  = None,
        deterministic: bool = True):
        
        self.set_training_mode(False)
        # observation, vectorized_env = self.obs_to_tensor(observation)
        obs = obs_as_tensor(obs, self.device)

        with th.no_grad():
            actions = self._predict(obs, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1,) + self.action_space.shape)
        actions = np.array([actions])
        return actions, state

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
        self.q_net.train(mode)
        self.training = mode


MlpPolicy = DQNPolicy


class CnnPolicy(DQNPolicy):
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
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
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


class MultiInputPolicy(DQNPolicy):
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
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
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