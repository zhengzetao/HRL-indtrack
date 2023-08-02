import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import gym
import time
import math
import torch
import wandb
import numpy as np
import pandas as pd
import torch as th
from torch import nn
from copy import deepcopy
from torch.nn import functional as F
from multiprocessing import Manager
import matplotlib.pyplot as plt
from DQN.policy import DQNPolicy
from MCTS.mcts import PortfolioSelector
# from stable_baselines3.common.buffers import ReplayBuffer
from tools.buffers import ReplayBuffer, MCTS_ReplayBuffer
from tools.api import ModelAPI
from tools.torch_layers import (
    # FlattenExtractor,
    # ScaleDotAttention,
    # SoftAttention,
    # MlpExtractor,
    # NatureCNN,
    create_mlp,
    RNN,
)
# from tools.neural_layers import FullyConnected,Neural_Net
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize, unwrap_vec_normalize
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common import utils
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update, get_schedule_fn, get_device, update_learning_rate
# from stable_baselines3.common.utils import explained_variance, set_random_seed, get_device, update_learning_rate, get_schedule_fn, obs_as_tensor, safe_mean
# from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
import ipdb



DQNSelf = TypeVar("DQNSelf", bound="DQN")


class High_level_policy():
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    # policy_aliases: Dict[str, Type[BasePolicy]] = {
    #     "MlpPolicy": MlpPolicy,
    #     "CnnPolicy": CnnPolicy,
    #     "MultiInputPolicy": MultiInputPolicy,
    # }

    def __init__(
        self,
        # policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 128,
        selected_num: int = 10,
        strategy: str="concat",
        tau: float = 0.005,
        vf_coef: float = 0.5,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 20,
        gradient_steps: int = 200,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        # optimize_memory_usage: bool = False,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        pre_trained_path: str = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.buffer_size = buffer_size
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.lookback = env.lookback
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.num_timesteps = 0
        self._episode_num = 0
        self._n_updates = 0
        self.gamma = gamma
        self.selected_num = int(selected_num)
        self.strategy = str(strategy)
        self.pre_trained_path = str(pre_trained_path)
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.action_noise = None
        self._last_obs = None
        self._custom_logger = False
        self.tensorboard_log = tensorboard_log
        self.learning_starts = learning_starts
        self.optimizer_class = optimizer_class
        # self.optimize_memory_usage = optimize_memory_usage
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        self.replay_buffer_kwargs={} if replay_buffer_kwargs is None else replay_buffer_kwargs
        self.Reward_total = []
        self.reward_episode = []
        self.train_freq = train_freq
        self.q_net, self.q_net_target = None, None
        self.seed = seed
        self.device = get_device(device)
        self.verbose = verbose
        self.tau = tau
        self.vf_coef = vf_coef

        if env is not None:
            env = self._wrap_env(env, self.verbose, monitor_wrapper=True)

        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.env = env
        self.n_envs = env.num_envs

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # super()._setup_model()
        # self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        #start lr=1e-4, end at lr=1e-6 when remaining process is 0.5
        self.lr_schedule = get_linear_fn(self.learning_rate, self.learning_rate*0.1, 1)
        
        # Use ReplayBuffer 
        self.replay_buffer_class = MCTS_ReplayBuffer
        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space[0],
            device=self.device,
            n_envs=self.n_envs,
            selected_num = self.selected_num,
            # optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs,
        )
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space[0],
            "selected_num": self.selected_num,
        }
        # wandb.init(config=self.model_kwargs,
        #        project="HRL-idxtrack",
        #        name="indtrack1:test1")

        # self.policy = DQNPolicy(  # DQNPolicy=MlpPolicy
        #     self.observation_space,
        #     self.action_space,
        #     self.lr_schedule,
        #     self.selected_num,
        #     self.strategy,
        #     **self.policy_kwargs,  # pytype:disable=not-instantiable
        # )
        # self.policy = self.policy.to(self.device)

        
        # play_worker = PortfolioSelector(config, cur_pipes, 0)
        # play_worker.start()
        # with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
        #     futures = []
        #     for i in range(config.play.max_processes):
        #         selector = PortfolioSelector(config, cur_pipes, i, use_history)
        #         logger.debug(f"Initialize Selector{i}...")
        #         futures.append(executor.submit(selector.start))

        # self.neural_model = FullyConnected(self.action_space.n, self.lookback)
        # self.neural_model = Neural_Net(self.action_space.n, self.lookback)
        self.neural_model = QNetwork(**self.net_args).cuda()
        # loading the weights from pre_train DQN
        self.neural_model.load_state_dict(th.load(self.pre_trained_path + f"policy.pth"))
        # self.neural_model = self.neural_model.to(self.device)
        # self.reg_loss = th.nn.L2Loss()
        self.optimizer = self.optimizer_class(self.neural_model.parameters(), lr=0.001, weight_decay=0.0001)
        # current_model, use_history = load_model(config)
        m = Manager()
        cur_pipes = m.list([self.neural_model.get_pipes() for _ in range(self.model_kwargs['max_processes'])])

        self.policy = PortfolioSelector(env=self.env, config=self.model_kwargs, K=self.selected_num, pipes=cur_pipes)
        # self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        # self._convert_train_freq()

        # self.q_net = self.policy.q_net
        # self.q_net_target = self.policy.q_net_target
        # # Copy running stats, see GH issue #996
        # self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        # self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # if self._n_calls % self.target_update_interval == 0:
        #     polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
        #     # Copy running stats, see GH issue #996
        #     polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self._logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        self.neural_model.train(True)
        # Update learning rate according to schedule
        update_learning_rate(self.optimizer, self.lr_schedule(self._current_progress_remaining))


        losses, policy_losses, value_losses = [], [], []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # with th.no_grad():
                # Compute the next Q-values using the target network
                # next_q_values = self.q_net_target(replay_data.next_observations)
                # sorted_q_values, indices = next_q_values.sort(dim=1, descending=True)
                # next_q_values = sorted_q_values[:,:self.selected_num]
            # print(replay_data.observations[:3,:,:5])

            # ini_input0 = th.zeros((1, 9, 31))
            # ini_input1 = th.ones((1, 9, 31))*10

            # ipdb.set_trace()

            # p1, v1 = self.neural_model(ini_input0)
            # p2, v2 = self.neural_model(ini_input1)
            # print(self.neural_model.input_fc[0].weight[0, :5])
            # print(ini_input0,p1)
            # print(ini_input1,p2)
            # exit()
            # print(type(replay_data.rewards))
            # print(type(replay_data.policies))
            pred_policy, pred_value = self.neural_model(replay_data.observations)
            target_value = replay_data.rewards
            target_policy = replay_data.policies
            # ipdb.set_trace()
            # print(pred_policy[:3,:10])
            # print(target_policy[:3,:])
            # print(replay_data.observations.shape,pred_policy.shape,target_policy.shape)
            # exit()
            # print(th.log(pred_policy)[0,:],target_policy[0,:])
                # Avoid potential broadcast issue
                # next_q_values = next_q_values.reshape(-1, 1)
                # next_q_values = next_q_values.reshape(-1, self.selected_num)
                # 1-step TD target
                # target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            policy_loss = -torch.mean(torch.sum(target_policy * torch.log(pred_policy),1))
            policy_losses.append(policy_loss.item())
            # 计算价值损失
            # value_loss = F.mse_loss(pred_value, target_value)
            # value_losses.append(value_loss.item())
            #计算L2正则化损失
            # reg_loss = self.reg(self.neural_model.parameters())
            # 计算总损失
            loss = policy_loss # + self.vf_coef * value_loss 
            # loss = policy_loss
            # print(loss)
            
            # current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            # current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            # current_q_values = current_q_values.sum(dim=1)
            
            # Compute Huber loss (less sensitive to outliers)
            # loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.optimizer.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.neural_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps
        if math.isnan(np.mean(losses)):
            ipdb.set_trace() 
        print("training loss:{}".format(np.mean(losses)))
        # wandb.log(
        #     {"policy_loss": np.mean(policy_losses),
        #     "value_loss": np.mean(value_losses),
        #     "final loss": np.mean(losses),
        #     "learning rate": self.optimizer.param_groups[0]['lr']},
        #     step=self.num_timesteps)
        self._logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self._logger.record("train/loss", np.mean(losses))

    # def train_PPO(self,):
    #     """
    #     Update policy using the currently gathered rollout buffer.
    #     """
    #     # Switch to train mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(True)
    #     # Update optimizer learning rate
    #     self._update_learning_rate(self.policy.optimizer)
    #     # Compute current clip range
    #     clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    #     # Optional: clip range for the value function
    #     if self.clip_range_vf is not None:
    #         clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    #     entrentropyopy_losses = []
    #     pg_losses, value_losses = [], []
    #     clip_fractions = []

    #     continue_training = True
    #     # train for n_epochs epochs
    #     for epoch in range(self.n_epochs):
    #         approx_kl_divs = []
    #         # Do a complete pass on the rollout buffer
    #         for rollout_data in self.rollout_buffer.get(self.batch_size):
    #             actions = rollout_data.actions
    #             if isinstance(self.action_space, spaces.Discrete):
    #                 # Convert discrete action from float to long
    #                 actions = rollout_data.actions.long().flatten()

    #             # Re-sample the noise matrix because the log_std has changed
    #             if self.use_sde:
    #                 self.policy.reset_noise(self.batch_size)

    #             values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
    #             values = values.flatten()
    #             # Normalize advantage
    #             advantages = rollout_data.advantages
    #             # Normalization does not make sense if mini batchsize == 1, see GH issue #325
    #             if self.normalize_advantage and len(advantages) > 1:
    #                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    #             # ratio between old and new policy, should be one at the first iteration
    #             ratio = th.exp(log_prob - rollout_data.old_log_prob)

    #             # clipped surrogate loss
    #             policy_loss_1 = advantages * ratio
    #             policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    #             policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

    #             # Logging
    #             pg_losses.append(policy_loss.item())
    #             clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
    #             clip_fractions.append(clip_fraction)

    #             if self.clip_range_vf is None:
    #                 # No clipping
    #                 values_pred = values
    #             else:
    #                 # Clip the difference between old and new value
    #                 # NOTE: this depends on the reward scaling
    #                 values_pred = rollout_data.old_values + th.clamp(
    #                     values - rollout_data.old_values, -clip_range_vf, clip_range_vf
    #                 )
    #             # Value loss using the TD(gae_lambda) target
    #             value_loss = F.mse_loss(rollout_data.returns, values_pred)
    #             value_losses.append(value_loss.item())

    #             # Entropy loss favor exploration
    #             if entropy is None:
    #                 # Approximate entropy when no analytical form
    #                 entropy_loss = -th.mean(-log_prob)
    #             else:
    #                 entropy_loss = -th.mean(entropy)

    #             entropy_losses.append(entropy_loss.item())

    #             loss = policy_loss + self.vf_coef * value_loss #+ self.ent_coef * entropy_loss 

    #             # Calculate approximate form of reverse KL Divergence for early stopping
    #             with th.no_grad():
    #                 log_ratio = log_prob - rollout_data.old_log_prob
    #                 approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
    #                 approx_kl_divs.append(approx_kl_div)

    #             if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
    #                 continue_training = False
    #                 if self.verbose >= 1:
    #                     print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
    #                 break

    #             # Optimization step
    #             self.policy.optimizer.zero_grad()
    #             loss.backward()
    #             # Clip grad norm
    #             th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #             self.policy.optimizer.step()

    #         self._n_updates += 1
    #         if not continue_training:
    #             break

    def predict(self, observation, num_step=10001, deterministic: bool = False):
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # if not deterministic and np.random.rand() < self.exploration_rate:
        #     unscaled_action = []
        #     while len(unscaled_action) < self.selected_num: 
        #         sample = self.action_space.sample()
        #         if sample not in unscaled_action: unscaled_action.append(sample) 
        #     # unscaled_action = np.array([unscaled_action])
        #     action = np.array([unscaled_action])
        # else:
            # action, state = self.policy.predict(observation, state, episode_start, deterministic)
        actions, policies = self.policy.select_portfolio(observation, num_step, training=False)
            # actions = np.array([action])
        return actions, policies

    def learn(
        self: DQNSelf,
        total_timesteps: int,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None, # for eval in the future
        # eval_freq: int = -1,
        # n_eval_episodes: int = 5,
        tb_log_name: str = "DQN",
        # eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):

        total_timesteps = self._setup_learn(
            total_timesteps,
            # eval_env,
            # callback,
            # eval_freq,
            # n_eval_episodes,
            # eval_log_path,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        times = time.localtime()
        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                # callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0:# and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                    # self.train_PPO()
                    # 一次episode数据训练完成，清空buffer
                    self.replay_buffer.reset()
                    print("One training session completed, buffer cleared!")

        # Plot the episode reward
        plt.plot(range(len(self.Reward_total)), self.Reward_total, "r")
        pd_reward_total = pd.DataFrame(data=self.Reward_total)
        pd_reward_total.to_csv("results/{}_{}_{}.episode_reward.csv".format(times.tm_hour,times.tm_min,times.tm_sec))
        plt.savefig("results/{}_{}_{}.episode_reward.png".format(times.tm_hour,times.tm_min,times.tm_sec))
        plt.close()

        return self

    def _setup_learn(
        self,
        total_timesteps: int,
        # eval_env: Optional[GymEnv],
        # callback: MaybeCallback = None,
        # eval_freq: int = 10000,
        # n_eval_episodes: int = 5,
        # log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int]:
        """
        Initialize different variables needed for training.
        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        # if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            # self.ep_info_buffer = deque(maxlen=100)
            # self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        # self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            # self._last_obs = self._last_obs[:,:,1:] #只取constituents变量出来，之前的obs里面包含index的变量
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            # if self._vec_normalize_env is not None:
            #     self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        return total_timesteps


    def collect_rollouts(
        self,
        env: VecEnv,
        # callback: BaseCallback,
        replay_buffer: MCTS_ReplayBuffer,
        train_freq: int =5,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(False)
        self.neural_model.train(False)
        # self.policy.train(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        # assert train_freq.frequency > 0, "Should at least collect one step or episode."

        # if env.num_envs > 1:
        #     assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        # if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
        #     action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        # if self.use_sde:
        #     self.actor.reset_noise(env.num_envs)

        # callback.on_rollout_start()
        continue_training = True
        dones = False

        # while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        while num_collected_steps < train_freq:
        # while not dones: #当采用episode结束就更新时
            # if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                # self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            # actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            
            # 这里调用MCTS搜索出K个节点，actions是长度为K的列表, policys是长度为K的列表，每个元素里面包含N长度的分布
            # observation = deepcopy(self._last_obs)
            actions, policys = self.policy.select_portfolio(self._last_obs, self.num_timesteps, training=True)
            # print(actions)
            
            # low-level policy: react according to the low-level obs
            # low_obs = self._last_obs[:,:,actions]
            # with th.no_grad():
            #     # Convert to pytorch tensor or to TensorDict
            #     obs_tensor = obs_as_tensor(low_obs, self.device)
                # actions, values, log_probs = self.policy(obs_tensor)
            # actions = actions.cpu().numpy()
            # if isinstance(self.action_space, spaces.Box):
            #     actions = np.clip(actions, self.action_space.low, self.action_space.high)


            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            if dones:
                self.Reward_total.append(sum(self.reward_episode))
                self.reward_episode = []
            else:
                self.reward_episode.append(rewards)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Retrieve reward and episode length if using Monitor wrapper
            # self._update_info_buffer(infos, dones)

            # Store high level data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, actions, policys, new_obs, rewards, dones, infos)
            # store low level data in low-level buffer
            # rollout_buffer.add(
            #     low_obs,  # type: ignore[arg-type]
            #     actions,
            #     rewards,
            #     self._last_episode_starts,  # type: ignore[arg-type]
            #     values,
            #     log_probs,
            # )
            # self._last_obs = new_obs
            # self._last_episode_starts = dones

            # with th.no_grad():
            # Compute value for the last timestep
            #   values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

            # rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

            # self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            # 是否更新targe_network为最新网络
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    # if log_interval is not None and self._episode_num % log_interval == 0:
                    #     self._dump_logs()

        # callback.on_rollout_end()
        # Print statement added to indicate completion of episode data collection
        print("Completed collection of one episode of data")

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


    # def _sample_action(
    #     self,
    #     learning_starts: int,
    #     action_noise: Optional[ActionNoise] = None,
    #     n_envs: int = 1,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Sample an action according to the exploration policy.
    #     This is either done by sampling the probability distribution of the policy,
    #     or sampling a random action (from a uniform distribution over the action space)
    #     or by adding noise to the deterministic output.
    #     :param action_noise: Action noise that will be used for exploration
    #         Required for deterministic policy (e.g. TD3). This can also be used
    #         in addition to the stochastic policy for SAC.
    #     :param learning_starts: Number of steps before learning for the warm-up phase.
    #     :param n_envs:
    #     :return: action to take in the environment
    #         and scaled action that will be stored in the replay buffer.
    #         The two differs when the action space is not normalized (bounds are not [-1, 1]).
    #     """
    #     # Select action randomly or according to policy
    #     # if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
    #     if self.num_timesteps < learning_starts:
    #         # 在神经网络没训练之前采用随机
    #         # Warmup phase
    #         # unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
    #         unscaled_action = []
    #         while len(unscaled_action) < self.selected_num: 
    #             sample = self.action_space.sample()
    #             if sample not in unscaled_action: unscaled_action.append(sample) 
    #         unscaled_action = np.array([unscaled_action])
    #     else:
    #         # 有了神经网络之后采用NN预测的值
    #         # Note: when using continuous actions,
    #         # we assume that the policy uses tanh to scale the action
    #         # We use non-deterministic action in the case of SAC, for TD3, it does not matter
    #         unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

    #     # Rescale the action from [low, high] to [-1, 1]
    #     if isinstance(self.action_space, gym.spaces.Box):
    #         scaled_action = self.policy.scale_action(unscaled_action)

    #         # Add noise to the action (improve exploration)
    #         if action_noise is not None:
    #             scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

    #         # We store the scaled action in the buffer
    #         buffer_action = scaled_action
    #         action = self.policy.unscale_action(scaled_action)
    #     else:
    #         # Discrete case, no need to normalize or clip
    #         buffer_action = unscaled_action
    #         action = buffer_action
    #     return action, buffer_action


    def _store_transition(
        self,
        replay_buffer: MCTS_ReplayBuffer,
        buffer_action: np.ndarray,
        policys: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        这个函数有很多不必要变量需要优化，例如_last_original_obs，new_obs_，_reward_都不必要，_vec_normalize_env是none
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).
        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            # 大概率不会被执行
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])
        node_state = self._last_original_obs
        for action, policy in zip(buffer_action[0], policys):
            replay_buffer.add(
                node_state,
                policy,
                action,
                reward_,
                dones,
                infos,
            )
            node_state[:,:,action] = 0

        # replay_buffer.add(
        #     self._last_original_obs,
        #     next_obs,
        #     buffer_action,
        #     reward_,
        #     dones,
        #     infos,
        # )

        # self._last_obs = new_obs[:,:,1:]
        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # if self._n_calls % self.target_update_interval == 0:
        #     polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            # polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self._logger.record("rollout/exploration_rate", self.exploration_rate)


    # def _excluded_save_params(self) -> List[str]:
    #     return super()._excluded_save_params() + ["q_net", "q_net_target"]

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space[0].seed(seed)
        if self.env is not None:
            self.env.seed(seed)

    def _wrap_env(self, env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose:
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])
        return env

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    # def save(self, path: str='./trained_models/SP500/') -> None:
    #     torch.save(self.policy.state_dict(), path + f"policy.pth")

    # def load(self, path: str='./trained_models/SP500/') -> None: 
    #     self.policy.load_state_dict(torch.load(path + f"policy.pth"))

    def save_model(path: str='./trained_models/SP500/'):
        return torch.save(self.policy.state_dict(), path + f"policy.pth")

    def load_model(self, path: str='./trained_models/SP500/'):
        use_history = False
        model = CChessModel(config)
        weight_path = config.resource.model_best_weight_path
        if not config_file:
            config_path = config.resource.model_best_config_path
            use_history = False
        else:
            config_path = os.path.join(config.resource.model_dir, config_file)
        if not load_model_weight(model, config_path, weight_path):
            model.build()
            save_as_best_model(model)
            use_history = True
        return model, use_history


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
        self.api = None
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
        self.stock_num = self.observation_space.shape[1]
        self.lookback = self.observation_space.shape[0]
        # q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        # self.q_net = nn.Sequential(*q_net)

    def _build_network(self, features_dim, action_dim, net_arch, activation_fn):
        # if self.strategy == "inter":
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
        self.encoder = nn.TransformerEncoderLayer(d_model=input_dim,nhead=2,batch_first=True,dropout=0.5,dim_feedforward=dim_feedforward)
        self.features_extractor = nn.Flatten()
        fuse_network = create_mlp(input_dim*action_dim, action_dim, net_arch, activation_fn)
        score_network = create_mlp(int(input_dim*1), action_dim, net_arch, activation_fn)
        self.fuse_network = nn.Sequential(*fuse_network)
        self.score_network = nn.Sequential(*score_network)

        # if self.strategy == "two":
        #     input_dim = self.observation_space.shape[0]
        #     print("execute the two-stream strategy")
        #     # encoder = LSHSelfAttention(dim = features_dim, heads = 2, bucket_size = 32, n_hashes = 6, causal = False)
        #     encoder = LSHAttention(bucket_size = 8, n_hashes = 6)
        #     num_head = 2
        #     num_layer = 2
        #     input_dim = self.observation_space.shape[0]
        #     dim_feedforward = 64
        #     transformer_args = {
        #     "num_layers": num_layer,
        #     "input_dim": self.observation_space.shape[0],
        #     "dim_feedforward": dim_feedforward,
        #     "num_heads": num_head,
        #     "dropout": 0.5,
        #     }
        #     self.stock_RNN = RNN(input_shape=1, hidden_dim=features_dim)
        #     self.index_RNN = RNN(input_shape=1, hidden_dim=features_dim)
        #     # self.encoder = nn.TransformerEncoderLayer(d_model=input_dim,nhead=2,batch_first=True,dropout=0.5,dim_feedforward=dim_feedforward)
        #     self.encoder = Autopadder(encoder)
        #     # self.encoder = LSHSA_Autopadder(encoder)
        #     self.features_extractor = nn.Flatten()
        #     # self.scale_dot_att = SoftAttention(input_dim)
        #     self.scale_dot_att = SoftAttention(features_dim)
        #     fuse_network = create_mlp(features_dim*action_dim, action_dim, net_arch, activation_fn)
        #     index_network = create_mlp(int(features_dim*action_dim), action_dim, net_arch, activation_fn)
        #     self.fuse_network = nn.Sequential(*fuse_network)
        #     self.index_network = nn.Sequential(*index_network)

    def extract_features(self, obs):
        # if self.strategy == "inter":

        # index_feat = index_feat.view(batch_size, 1, int(self.features_dim/2))
        # stock_feat = stock_feat.view(batch_size, stock_num, int(self.features_dim/2))
        # fuse_feat = torch.cat((index_feat,stock_feat),1)
        # features = self.features_extractor(fuse_feat)
        # features = self.fuse_network(features)
        obs = obs.reshape(-1, self.stock_num, self.lookback).cuda()
        # obs = obs.permute(0, 2, 1)
        stock_tran_hidden = self.encoder(obs)
        stock_tran_hidden = stock_tran_hidden[:,1:,:]
        index_feat = stock_tran_hidden[:,0,:]
        flatten_feat = self.features_extractor(stock_tran_hidden)
        stock_score = self.fuse_network(flatten_feat)
        index_score = self.score_network(index_feat)
        features = stock_score * index_score

        # if self.strategy == "two":
        #     obs = obs.permute(0, 2, 1)
        #     stock_data = obs[:,1:,:]
        #     index_data = obs[:,0,:]
        #     lookback = obs.shape[2]
        #     batch_size = obs.shape[0]
        #     stock_num = stock_data.shape[1]
        #     index_rnn_hidden = self.index_RNN(index_data.reshape(batch_size, lookback))
        #     index_rnn_hidden = index_rnn_hidden[:,-1,:].view(batch_size, self.features_dim)
        #     stock_rnn_hidden = self.stock_RNN(stock_data.reshape(batch_size*stock_num, lookback))
        #     stock_rnn_hidden = stock_rnn_hidden[:,-1,:].view(batch_size*stock_num, self.features_dim)
        #     stock_data = stock_rnn_hidden.view(batch_size, stock_num, self.features_dim)
        #     index_data = index_rnn_hidden.view(batch_size, self.features_dim)

        #     stock_reform_hidden = self.encoder(stock_data, stock_data)
        #     # stock_reform_hidden = self.encoder(stock_data)
        #     (stock_index_hidden,att) = self.scale_dot_att(index_data,stock_data,stock_data)
        #     flatten_stock_feat1 = self.features_extractor(stock_reform_hidden) #stocks feat with relation among assets
        #     flatten_stock_feat2 = self.features_extractor(stock_index_hidden)  #stocks feat relation between index and assets
        #     stock_score = self.fuse_network(flatten_stock_feat1)
        #     index_score = self.index_network(flatten_stock_feat2)
        #     features = 0.9*stock_score + 0.1*index_score

        return features

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # if self.strategy == "concat":
        #     return self.q_net(self.extract_features(obs))
        # if self.strategy == "solo" or self.strategy =="inter" or self.strategy=="two":
        q_values = self.extract_features(obs)
        pi = th.nn.Softmax(dim=1)(q_values)
        v = torch.zeros([q_values.shape[0],1])
        return pi, v

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        k_actions = q_values.argsort(dim=1, descending=True)[:,:self.selected_num]
        # action = q_values.argmax(dim=1).reshape(-1)
        return k_actions

    def get_pipes(self, num=1, api=None, need_reload=True):
        if self.api is None:
            self.api = ModelAPI(self)
            self.api.start(need_reload)
        return self.api.get_pipe(need_reload)

    def close_pipes(self):
        if self.api is not None:
            self.api.close()
            self.api = None
