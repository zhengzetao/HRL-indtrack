import json
import logging
import os
import time
from argparse import ArgumentParser
import datetime

import config


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    parser.add_argument('--dataset',default='indtrack1', help="indtrack1-7")
    parser.add_argument('--K',default=10, help="10,50,90")
    parser.add_argument('--strategy',default='inter', help="concat,solo")
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    if options.mode == "train":
        if options.dataset == 'SP500':
            import train_SP500
            train_SP500.train_stock_trading(options.dataset,options.K,options.strategy)
        else:
            import train
            train.train_stock_trading(options.dataset,options.K,options.strategy)
        
        
if __name__ == "__main__":
    main()
