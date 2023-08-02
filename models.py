# common library
import pandas as pd
import numpy as np
import torch as th
import time
import gym


import config
from preprocessor.preprocessors import FeatureEngineer, data_split, series_decomposition

from env.env_portfolio import StockPortfolioEnv
from MCTS.policy import High_level_policy as MCTS
from A2C.a2c import A2C
from DQN.dqn import DQN
from tools.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
# from stable_baselines3.common.vec_env import DummyVecEnv


MODELS = {"a2c": A2C, "dqn": DQN, "mcts":MCTS} #can be extended with other RL algos


MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        train_PPO()
            the implementation for PPO algorithm
        train_A2C()
            the implementation for A2C algorithm
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    @staticmethod
    def DRL_prediction(model, environment):
        test_env, test_state = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        truth_account_memory = []
        error_memory = []
        episode_start = 1
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _ = model.predict(test_state, deterministic=True)
            # test_state, rewards, dones, info = test_env.step([action])
            test_state, rewards, dones, info = test_env.step(action)
            episode_start = dones
            if i == (len(environment.df.index.unique()) - 2):
              account_memory = test_env.env_method(method_name="save_asset_memory")
              truth_account_memory = test_env.env_method(method_name="save_truth_asset_memory")
              error_memory = test_env.env_method(method_name="save_tracking_error_memory")
              # days_return_memory = test_env.env_method(method_name="save_daily_return_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], truth_account_memory[0], error_memory[0]#, days_return_memory[0]

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        selected_num,
        strategy=None,
        pre_trained_path=None,
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        model = MODELS[model_name](
            env=self.env,
            selected_num=selected_num,
            strategy=strategy,
            pre_trained_path=pre_trained_path,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            model_kwargs=model_kwargs,
        )
        return model

    def train_model(self, model, tb_log_name, total_timesteps=5000, eval_env=None):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, eval_env=eval_env)
        # model.save(config.TRAINED_MODEL_DIR)
        return model

