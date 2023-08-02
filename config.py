import pathlib

# import finrl

import pandas as pd
import datetime
import os


TRAINED_MODEL_DIR = f"trained_models"
# DATASET_DIR = PACKAGE_ROOT / "data"

# data
# TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
# TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
# TESTING_DATA_FILE = "test.csv"

# now = datetime.datetime.now()
# TRAINED_MODEL_DIR = f"trained_models/{now}"
DATA_SAVE_DIR = f"datasets"
TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_LOG_DIR = f"tensorboard_log"
RESULTS_DIR = f"results"
# os.makedirs(TRAINED_MODEL_DIR)


## time_fmt = '%Y-%m-%d'
START_DATE = "2009-01-01"
END_DATE = "2021-07-01"

START_TRADE_DATE = "2020-07-01"
SELECTED_STOCK_NUM = 10

## dataset default columns
DEFAULT_DATA_COLUMNS = ["date", "tic", "close"]

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
#TECHNICAL_INDICATORS_LIST = ["macd","boll_ub","boll_lb","rsi_30", "cci_30", "dx_30","close_30_sma","close_60_sma"]
TECHNICAL_INDICATORS_LIST = ["macd","rsi_30", "close_5_sma", "close_15_sma", "close_30_sma"]


## Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
DQN_PARAMS = {
    "buffer_size": 1000000, 
    "learning_rate": 0.0002,
    "learning_starts": 3000,
}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": "auto_0.1",
}
MCTS_PARAMS = {
    "max_processes":1,
    "search_threads":50,
    "virtual_loss": 10,
    "simulation_num_per_move_train": 50,
    "simulation_num_per_move_test": 200,
    "c_puct": 2.5,
    "noise_eps": 0.15,
    "dirichlet_alpha": 0.2,
    "tau_decay_rate": 0.9,
}

########################################################

# defined the timestamp for training on sp500, with 6 months for training
TRAIN_TIMESTAMP_6 = [
		"20060630", "20060929", "20061229",
        "20070330", "20070629", "20070928", "20071231",
        "20080331", "20080630", "20080930", "20081231",
        "20090331", "20090630", "20090930", "20091231",
        "20100331", "20100630", "20100930", "20101231",
        "20110331", "20110630", "20110930", "20111230",
        "20120330", "20120629", "20120928", "20121231",
        "20130328", "20130628", "20130930", "20131231",
        "20140331", "20140630", "20140930", "20141231",
        "20150331", "20150630", "20150930", "20151231",
        "20160331", "20160630", "20160930", "20161230",
        "20170331", "20170630", "20170929", "20171229",
        "20180329", "20180629", "20180928", "20181031"
]

# defined the timestamp for training on sp500, with 12 months for training
TRAIN_TIMESTAMP_12 = ["20061229",
        "20070330", "20070629", "20070928", "20071231",
        "20080331", "20080630", "20080930", "20081231",
        "20090331", "20090630", "20090930", "20091231",
        "20100331", "20100630", "20100930", "20101231",
        "20110331", "20110630", "20110930", "20111230",
        "20120330", "20120629", "20120928", "20121231",
        "20130328", "20130628", "20130930", "20131231",
        "20140331", "20140630", "20140930", "20141231",
        "20150331", "20150630", "20150930", "20151231",
        "20160331", "20160630", "20160930", "20161230",
        "20170331", "20170630", "20170929", "20171229",
        "20180329", "20180629", "20180928", "20181031"
]

