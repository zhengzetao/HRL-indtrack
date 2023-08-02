import os
import sys
sys.path.append("..")
import config
import yfinance as yf
import pandas as pd

data_df = pd.DataFrame()
ticker_list=config.DOW_30_TICKER
ticker_list = config.HSI_50_TICKER
if not os.path.exists("./datasets/HSI_50_TICKER"):
    os.makedirs("./datasets/HSI_50_TICKER")
for tic in ticker_list:
    temp_df = yf.download(tic, start=config.START_DATE, end=config.END_DATE)
    temp_df["tic"] = tic
    print(temp_df)
    # data_df = data_df.append(temp_df)
    # print(data_df.head(5))
	# data.to_csv('./datasets/HSI_50_TICKER/'+tic+'.csv')
	# print('save {} successfully'.format(tic))
