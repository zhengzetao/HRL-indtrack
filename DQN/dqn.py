import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import gym
import time
import ipdb
import torch
import numpy as np
import pandas as pd
import torch as th
from copy import deepcopy
from torch.nn import functional as F
import matplotlib.pyplot as plt
from DQN.policy import DQNPolicy
from PPO.policy import ActorCriticPolicy
from PPO.ppo import PPO
# from stable_baselines3.common.buffers import ReplayBuffer
from tools.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize, unwrap_vec_normalize
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common import utils
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update, get_schedule_fn, get_device, update_learning_rate, obs_as_tensor
# from stable_baselines3.common.utils import explained_variance, set_random_seed, get_device, update_learning_rate, get_schedule_fn, obs_as_tensor, safe_mean
# from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy

DQNSelf = TypeVar("DQNSelf", bound="DQN")


class DQN():
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
        batch_size: int = 32,
        selected_num: int = 10,
        strategy: str="concat",
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 15,
        gradient_steps: int = 1,
        # replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        pre_trained_path: str = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.buffer_size = model_kwargs['buffer_size'] if model_kwargs['buffer_size'] is not None else buffer_size  
        self.observation_space = env.observation_space
        self.action_space = env.action_space
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
        self.learning_rate =  model_kwargs['learning_rate'] if model_kwargs['learning_rate'] is not None else learning_rate 
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
        self.learning_starts = model_kwargs['learning_starts'] if model_kwargs['learning_starts'] is not None else learning_starts
        # self.optimize_memory_usage = optimize_memory_usage
        # self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.replay_buffer_kwargs={} if replay_buffer_kwargs is None else replay_buffer_kwargs
        self.Reward_total = []
        self.reward_episode = []
        self.train_freq = train_freq
        self.n_steps = 15
        self.q_net, self.q_net_target = None, None
        self.seed = seed
        self.device = get_device(device)
        self.verbose = verbose
        self.tau = tau

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
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        # Use ReplayBuffer 
        # self.replay_buffer_class = ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space[0],
            device=self.device,
            n_envs=self.n_envs,
            selected_num = self.selected_num,
            # optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs,
        )
        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            # self.observation_space,
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space.shape[0], self.selected_num)),
            self.action_space[1],
            self.device,
            # gamma=self.gamma,
            # gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.dqn_policy = DQNPolicy(  # DQNPolicy=MlpPolicy
            self.observation_space,
            self.action_space[0], # high-level action space
            self.lr_schedule,
            self.selected_num,
            self.strategy,
            # **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        
        self.ppo_policy = ActorCriticPolicy(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space[1],  # low-level action space
            self.lr_schedule,
            self.selected_num,
            # **self.policy_kwargs  # pytype:disable=not-instantiable
        )

        self.PPO = PPO(
            env = self.env
        )
        
        self.dqn_policy = self.dqn_policy.to(self.device)
        self.ppo_policy = self.ppo_policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        # self._convert_train_freq()

        self.q_net = self.dqn_policy.q_net
        self.q_net_target = self.dqn_policy.q_net_target
        # self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
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

    # def _create_aliases(self) -> None:
    #     self.q_net = self.dqn_policy.q_net
    #     self.q_net_target = self.dqn_policy.q_net_target

    # def _on_step(self) -> None:
    #     """
    #     Update the exploration rate and target network if needed.
    #     This method is called in ``collect_rollouts()`` after each step in the environment.
    #     """
    #     self._n_calls += 1
    #     if self._n_calls % self.target_update_interval == 0:
    #         polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
    #         # Copy running stats, see GH issue #996
    #         polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

    #     self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
    #     self._logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.dqn_policy.set_training_mode(True)
        # Update learning rate according to schedule
        # self._update_learning_rate(self.dqn_policy.optimizer)
        update_learning_rate(self.dqn_policy.optimizer, self.lr_schedule(self._current_progress_remaining))

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                # next_q_values, _ = next_q_values.max(dim=1)
                sorted_q_values, indices = next_q_values.sort(dim=1, descending=True)
                next_q_values = sorted_q_values[:,:self.selected_num]
                # next_q_values = next_q_values.sum(dim=1) #
                
                # Avoid potential broadcast issue
                # next_q_values = next_q_values.reshape(-1, 1)
                next_q_values = next_q_values.reshape(-1, self.selected_num) #
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            # current_q_values = current_q_values.sum(dim=1) #
            
            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.dqn_policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.dqn_policy.parameters(), self.max_grad_norm)
            self.dqn_policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self._logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self._logger.record("train/loss", np.mean(losses))

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
    ) -> DQNSelf:

        total_timesteps = self._setup_learn(total_timesteps, reset_num_timesteps, tb_log_name, progress_bar)
        times = time.localtime()
        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                # callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                rollout_buffer=self.rollout_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            self.PPO.train(self.rollout_buffer, self._current_progress_remaining)
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

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
        replay_buffer: ReplayBuffer,
        rollout_buffer: RolloutBuffer,
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
        self.dqn_policy.set_training_mode(False)
        # self.dqn_policy.train(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        # assert train_freq.frequency > 0, "Should at least collect one step or episode."

        # if env.num_envs > 1:
        #     assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        # if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
        #     action_noise = VectorizedActionNoise(action_noise, env.num_envs)
        rollout_buffer.reset()
        # if self.use_sde:
        #     self.actor.reset_noise(env.num_envs)

        # callback.on_rollout_start()
        continue_training = True

        # while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        while num_collected_steps < train_freq:
            # if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                # self.actor.reset_noise(env.num_envs)
            self.dqn_policy.set_training_mode(False)
            self.ppo_policy.set_training_mode(False)
            
            # actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            # high_action, _ = self.predict(self._last_obs) # self._last_obs shape [1, T, N]
            # high_action, _ = self.dqn_policy.predict(self._last_obs) # self._last_obs shape [1, T, N]
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                high_action = self.dqn_policy(obs_tensor)
                low_action, values, log_probs = self.ppo_policy(obs_tensor[:,:,high_action[0]+1])
            high_action = high_action.cpu().numpy()
            low_action = low_action.cpu().numpy()
            clipped_low_action = np.clip(low_action, self.action_space[1].low, self.action_space[1].high)
            actions = np.zeros((1, self.action_space[0].n))
            for k, v in enumerate(high_action[0]): actions[0][v] = clipped_low_action[0][k]
            # with th.no_grad():
            #     high_action = self.dqn_policy(self._last_obs)
            #     high_action = high_action.cpu().numpy()
                
            #     low_action, values, log_probs = self.ppo_policy(self._last_obs[:,:,high_action[0]+1])
            #     low_action = low_action.cpu().numpy()
            #     clipped_low_action = np.clip(low_action, self.action_space[1].low, self.action_space[1].high)
            #     actions = np.zeros(self.action_space[0].n)
            #     for k, v in enumerate(high_action[0]): actions[v] = clipped_low_action[0][k]
            #     actions = np.array(act)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(high_action+1)

            if dones:
                self.Reward_total.append(sum(self.reward_episode))
                self.reward_episode = []
            else:
                self.reward_episode.append(rewards)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            # callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            # if callback.on_step() is False:
            #     return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            # self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            # self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
            replay_buffer.add(self._last_obs, new_obs, high_action, rewards, dones, infos)
            rollout_buffer.add(self._last_obs[:,:,high_action[0]+1], low_action, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

            # self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
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

        with th.no_grad():
            # Compute value for the last timestep
            new_obs_tensor = obs_as_tensor(new_obs, self.device)
            high_action = self.dqn_policy(new_obs_tensor)
            values = self.ppo_policy.predict_values(new_obs_tensor[:,:,high_action[0]+1])

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        # if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
        if self.num_timesteps < learning_starts:
            # Warmup phase
            # unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            unscaled_action = []
            while len(unscaled_action) < self.selected_num: 
                sample = self.action_space[0].sample()
                if sample not in unscaled_action: unscaled_action.append(sample) 
            unscaled_action = np.array([unscaled_action])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        # if isinstance(self.action_space, gym.spaces.Box):
        #     scaled_action = self.dqn_policy.scale_action(unscaled_action)

        #     # Add noise to the action (improve exploration)
        #     if action_noise is not None:
        #         scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

        #     # We store the scaled action in the buffer
        #     buffer_action = scaled_action
        #     action = self.dqn_policy.unscale_action(scaled_action)
        # else:
            # Discrete case, no need to normalize or clip
        buffer_action = unscaled_action
        action = buffer_action
        return action, buffer_action

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            # if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
            #     if isinstance(self.observation_space, gym.spaces.Dict):
            #         n_batch = observation[list(observation.keys())[0]].shape[0]
            #     else:
            #         n_batch = observation.shape[0]
            #     action = np.array([self.action_space.sample() for _ in range(n_batch)])
            # else:
            #     action = np.array(self.action_space.sample())
            unscaled_action = []
            while len(unscaled_action) < self.selected_num: 
                sample = self.action_space[0].sample()
                if sample not in unscaled_action: unscaled_action.append(sample) 
            # unscaled_action = np.array([unscaled_action])
            action = np.array([unscaled_action])
        else:
            action, state = self.dqn_policy.predict(observation, state, episode_start, deterministic)
            # action = np.array([action])
        return action, state

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        这儿的_last_obs表current_obs,new_obs表next_obs与_last_original_obs表示混乱
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
        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                next_obs[i] = infos[i]["terminal_observation"]

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs


    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

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

    def save(self, path: str='./trained_models/SP500/') -> None:
        torch.save(self.dqn_policy.q_net.state_dict(), path + f"policy.pth")

    def load(self, path: str='./trained_models/SP500/') -> None: 
        self.dqn_policy.load_state_dict(torch.load(path + f"policy.pth"))