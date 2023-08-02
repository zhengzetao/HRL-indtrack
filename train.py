import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime

import config
from preprocessor.dataloader import DataLoader
from preprocessor.preprocessors import FeatureEngineer, data_split, series_decomposition
# from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from env.env_portfolio import StockPortfolioEnv
from models import DRLAgent
from plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, convert_daily_return_to_pyfolio_ts

import itertools


def train_stock_trading(dataset, K, strategy):
    """
    train an agent
    """

    print("==============Start Fetching Data===========")
    df, indexing_data = DataLoader(
        portfolio_name=dataset,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        # ticker_list=Ticker_list,
    ).fetch_data()

    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=False,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False,
    )

    df = fe.preprocess_data(df)

    # add covariance matrix as states
    df=df.sort_values(['Week_ID','tic'],ignore_index=True)
    df.index = df.Week_ID.factorize()[0]
    return_list, price_list = [], []

    # look back is ten week
    lookback=9
    for i in range(lookback,len(df.index.unique())-1):
      data_lookback = df.loc[i-lookback:i,:]
      index_lookback = indexing_data.loc[i-lookback:i,:]
      price_lookback=data_lookback.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close')
      price_list.append(price_lookback.pct_change().fillna(method='backfill').values)    # price_list shape (lookback, stock_num)
      return_lookback = df.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close').pct_change().dropna()
      return_list.append(return_lookback.values)
      # return_lookback = index_lookback.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close').pct_change().dropna()
      # index_value_list.append(index_lookback['Indexing'].values)

    df_fuse = pd.DataFrame({'Week_ID':df.Week_ID.unique()[lookback:-1],'price_list':price_list,'return_list':return_list})
    df = df.merge(df_fuse, on='Week_ID')
    df = df.sort_values(['Week_ID','tic']).reset_index(drop=True)

    truth_indexing = indexing_data['Indexing'].pct_change().dropna()
    indexing_data = indexing_data['Indexing'].pct_change().dropna().iloc[lookback:]
    

    # Training & Trading data split
    # train = data_split(df, config.START_DATE, config.START_TRADE_DATE)
    # trade = data_split(df, config.START_TRADE_DATE, config.END_DATE)
    train, trade = data_split(df, indexing_data, lookback)

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        # "action_space": config.SELECTED_STOCK_NUM,
        "action_space": stock_dimension-1, # one is index 
        "reward_scaling": 1e-4,
        "lookback": lookback+1, #lookback=9, but the data length is 10
        "selected_num": K,
        }
    TRAINED_MODEL_PATH = "./" + config.TRAINED_MODEL_DIR  + "/" + dataset + "/"

    e_train_gym = StockPortfolioEnv(df=train, indexing=truth_indexing, **env_kwargs)
    e_trade_gym = StockPortfolioEnv(df=trade, indexing=truth_indexing, turbulence_threshold=250, **env_kwargs)
    # env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=e_train_gym)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    ################ A2C #########################
    # model_a2c = agent.get_model("a2c",K)
    # trained_a2c = agent.train_model(
    #     model=model_a2c, tb_log_name="a2c", total_timesteps=2000, eval_env = e_trade_gym
    # )

    ################ DQN #########################
    model_dqn = agent.get_model("dqn", K, strategy=strategy)
    trained_dqn = agent.train_model(
        model=model_dqn, tb_log_name="dqn", total_timesteps=50000, eval_env = e_trade_gym
    )
    model_dqn.save(TRAINED_MODEL_PATH)

    print("============== Heuristic-guided Search with MCTS===========")
    model_mcts = agent.get_model("mcts", K, pre_trained_path=TRAINED_MODEL_PATH)
    trained_mcts = agent.train_model(
        model=model_mcts, tb_log_name="mcts", total_timesteps=150, eval_env = e_trade_gym
    )

    print("============== Start Trading WITH DQN===========")
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    df_account_value, df_actions, df_error= DRLAgent.DRL_prediction(
        model=trained_dqn, environment = e_trade_gym
    )

    print("============== Start Trading WITH MCTS===========")
    df_account_value, df_actions, df_error= DRLAgent.DRL_prediction(
        model=trained_mcts, environment = e_trade_gym
    )
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + dataset + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + dataset + ".csv")
    df_error.to_csv("./" + config.RESULTS_DIR + "/df_error_" + dataset + ".csv")

    print("==============Get Backtest Results===========")
 
    from pyfolio import timeseries
    DRL_strat = convert_daily_return_to_pyfolio_ts(df_account_value)
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func( returns=DRL_strat, 
                              factor_returns=DRL_strat, 
                                positions=None, transactions=None, turnover_denom="AGB")
    print("==============DRL Strategy Stats===========")
    print(perf_stats_all)
