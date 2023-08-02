import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from preprocessor.dataloader import DataLoader
import pywt
import itertools
import sys
sys.path.append("..")
import config



def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def data_split_SP500(df, indexing, lookback):
    """
    split the dataset into slices, and each slice contains training and testing part
    :param data: (df) pandas dataframe, lookback(to judge the half or whole year as input)
    :return: (df) pandas dataframe
    """
    if lookback<250:
        timestamp = config.TRAIN_TIMESTAMP_6
        # 10 season for training (the season for training totally 12, another 2 season as input), 1 for testing
        time_gap = 11 
    else:
        timestamp = config.TRAIN_TIMESTAMP_12
        time_gap = 9  # 9  season
    df['Date'] = pd.to_datetime(df['Date'].apply(str), errors='coerce')
    indexing['Date'] = pd.to_datetime(indexing['Date'].apply(str), errors='coerce')
    # return_lookback = df.pivot_table(index = 'Date',columns = 'tic', values = 'Close').pct_change().fillna(0)

    train, test, index_train, index_test= [], [], [], []
    k, j= 0, 0
    for _ in range(len(timestamp) - time_gap):
        train_return_list, test_return_list, train_index, test_index = [], [], [], []
        train_test = timestamp[k:k+time_gap]        
        train_ts_slice = train_test[:-1]
        test_ts_slice = train_test[-1:]
        train_slice = df[df.Date.isin(pd.to_datetime(train_ts_slice))]
        test_slice = df[df.Date.isin(pd.to_datetime(test_ts_slice))]
        train_slice = train_slice.sort_values(['Week_ID','tic']).reset_index(drop=True)
        test_slice = test_slice.sort_values(['Week_ID','tic']).reset_index(drop=True)
        train_slice.index = train_slice.Week_ID.factorize()[0]
        test_slice.index = test_slice.Week_ID.factorize()[0]

        # obtain the return of stocks
        retrun_train_test = timestamp[k:k+time_gap+1]
        train_return_slice = retrun_train_test[:-1]
        test_return_slice = retrun_train_test[-1:]
        for i in range(len(train_return_slice)-1):
            condition1 = df['Date']>=pd.to_datetime(train_return_slice[i])
            condition2 = df['Date']<=pd.to_datetime(train_return_slice[i+1])
            condition1_index = indexing['Date']>=pd.to_datetime(train_return_slice[i])
            condition2_index = indexing['Date']<=pd.to_datetime(train_return_slice[i+1])
            tra_stock_data = df[condition1 & condition2]
            tra_index_data = indexing[condition1_index & condition2_index]
            tra_stock_data = tra_stock_data.sort_values(['Week_ID','tic']).reset_index(drop=True)
            tra_stock_data.index = tra_stock_data.Week_ID.factorize()[0]
            train_return_list.append(tra_stock_data.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close').pct_change().dropna().values)
            train_index.append(tra_index_data['Close'].pct_change().dropna().values)
        condition3 = df['Date']>=pd.to_datetime(train_return_slice[-1])
        condition4 = df['Date']<=pd.to_datetime(test_return_slice[-1])
        condition3_index = indexing['Date']>=pd.to_datetime(train_return_slice[-1])
        condition4_index = indexing['Date']<=pd.to_datetime(test_return_slice[-1])
        test_stock_data = df[condition3 & condition4]
        test_stock_data = test_stock_data.sort_values(['Week_ID','tic']).reset_index(drop=True)
        test_stock_data.index = test_stock_data.Week_ID.factorize()[0]
        test_return_list.append(test_stock_data.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close').pct_change().dropna().values)
        test_index_data = indexing[condition3_index & condition4_index]
        test_index.append(test_index_data['Close'].pct_change().dropna().values)

        # print(train_slice.Date.unique()[:],test_slice.Date.unique())

        train_temp_df = pd.DataFrame({'Week_ID':train_slice.Week_ID.unique(),'return_list':train_return_list})
        test_temp_df = pd.DataFrame({'Week_ID':test_slice.Week_ID.unique(),'return_list':test_return_list})
        train_index_df = pd.DataFrame({'Week_ID':train_slice.Week_ID.unique(),'Date':train_slice.Date.unique(),'index_list':train_index})
        test_index_df = pd.DataFrame({'Week_ID':test_slice.Week_ID.unique(),'Date':test_slice.Date.unique(),'index_list':test_index})
        train_slice = train_slice.merge(train_temp_df, on='Week_ID')
        test_slice = test_slice.merge(test_temp_df, on='Week_ID')
        train_slice = train_slice.sort_values(['Week_ID','tic']).reset_index(drop=True)
        test_slice = test_slice.sort_values(['Week_ID','tic']).reset_index(drop=True)
        train_slice.index = train_slice.Week_ID.factorize()[0]
        test_slice.index = test_slice.Week_ID.factorize()[0]

        # add the rubbish data at the last line
        train_slice = add_rubbish_line(train_slice)
        test_slice = add_rubbish_line(test_slice)

        # print(np.array(train_index_df.loc[0].index_list).shape,np.array(train_slice.loc[0].return_list.values[0]).shape)
        # print(np.array(test_index_df.loc[0].index_list).shape,np.array(test_slice.loc[0].return_list.values[0]).shape)

        train.append(train_slice)
        test.append(test_slice)
        index_train.append(train_index_df)
        index_test.append(test_index_df)

        k += 1
        j += 1

    # j = 0
    # for _ in range(len(timestamp) - time_gap):
    #     condition1 = df['Date']>pd.to_datetime(timestamp[j])
    #     condition2 = df['Date']<=pd.to_datetime(timestamp[j+1])
    #     data = df[condition1 & condition2]
    #     stocks_return = data.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close').pct_change().dropna()
    #     return_list.append(stocks_return.values)
    #     # print(stocks_return.values.shape)
    #     j += 1 
    # exit()

    # df = df.sort_values(["Date", "tic"], ignore_index=True)
    # df.index = df.Date.factorize()[0]
    return train, test, index_train, index_test

def add_rubbish_line(df):
    # 在dataframe最后增加一行无用数据,通过复制前一行数据，此做法仅在SP500数据集上使用，目的是为env中确定terminal而用
    import warnings
    warnings.filterwarnings('ignore')

    data_add = df.loc[df.index[-1]]
    data_add['Week_ID'] = data_add['Week_ID'] + 1
    data_add['Date'] = data_add['Date'] + pd.DateOffset(1)
    data_add.index = data_add.index+1
    df = df.append(data_add)
    return df

def data_split(df, indexing, lookback):
    """
    split the dataset into training and testing set
    :param data: (df) pandas dataframe, indexing value.
    :return: (df) pandas dataframe
    """
    train_data = df[df.Week_ID <= 145]
    # train_index = indexing[indexing.index <= 146]
    train_data = train_data.sort_values(["Week_ID", "tic"], ignore_index=True)
    train_data.index = train_data.Week_ID.factorize()[0]
    
    test_data = df[df.Week_ID > 145]
    # test_index = indexing[indexing.index > 146]
    test_data = test_data.sort_values(["Week_ID", "tic"], ignore_index=True)
    test_data.index = test_data.Week_ID.factorize()[0]
    
    return train_data, test_data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


def series_decomposition(data, level):
    '''
    reference: https://www.freesion.com/article/2882783957/

    decompose the close price series into multi-level series 
    using haar decomposition
    input param: data.shape=[lookback, stocks_num], max decompose level
    output param: decomposed array
    '''
    dec_list = [[] for i in range(level+1)]
    wavelet = 'haar'
    for i in range(data.shape[1]):
        coeffs =pywt.wavedec(data[:,i], wavelet, level=level)
        level_id = np.eye(level+1)
        for j, coeff in enumerate(coeffs):
            if level == 1:
                rec_coefs = []
                level_id_list = [[item] for item in level_id[j]]
                temp_coefs = np.multiply(coeffs,level_id_list).tolist()
                for coef in temp_coefs:
                    rec_coefs.append(np.array(coef))
            else:
                rec_coefs = np.multiply(coeffs,level_id[j]).tolist()
            # temp = pywt.waverec(np.multiply(coeffs,level_id[j]).tolist(),wavelet)
            temp = pywt.waverec(rec_coefs, wavelet)
            dec_list[j].append(temp.astype(np.float32))

    '''display the multi-level series'''
    # import matplotlib.pyplot as plt
    # X = range(data.shape[0])
    # plt.figure(figsize=(12, 12))
    # plt.subplot(611)
    # plt.plot(X, dec_list[0][1])
    # plt.title('A3')
    # plt.subplot(612)
    # plt.plot(X, dec_list[1][1])
    # plt.title('D3')
    # plt.subplot(613)
    # plt.plot(X, dec_list[2][1])
    # plt.title('D2')
    # plt.subplot(614)
    # plt.plot(X, dec_list[3][1])
    # plt.title('D1')
    # # plt.plot(self.portfolio_return_memory, "r")
    # plt.savefig("results/decompose.png")
    # plt.close()

    return np.array(dec_list)

class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_vix = False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        #clean data
        df = self.clean_data(df)
        
        # add technical indicators using stockstats
        if self.use_technical_indicator == True:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")
            
        # add vix for multiple stock
        if self.use_vix == True:
            df = self.add_vix(df)
            print("Successfully added vix")
            
        # add turbulence index for multiple stock
        if self.use_turbulence == True:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature == True:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df
    
    def clean_data(self, data):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step 
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df=df.sort_values(['Week_ID','tic'],ignore_index=True)
        df.index = df.Week_ID.factorize()[0]
        merged_closes = df.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close')
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        #df = data.copy()
        #list_ticker = df["tic"].unique().tolist()
        #only apply to daily level data, need to fix for minute level
        #list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
        #combination = list(itertools.product(list_date,list_ticker))

        #df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
        #df_full = df_full[df_full['date'].isin(df['date'])]
        #df_full = df_full.sort_values(['date','tic'])
        #df_full = df_full.fillna(0)
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=['tic','Week_ID'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator['tic'] = unique_ticker[i]
                    temp_indicator['Week_ID'] = df[df.tic == unique_ticker[i]]['Week_ID'].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[['tic','Week_ID',indicator]],on=['tic','Week_ID'],how='left')
        df = df.sort_values(by=['Week_ID','tic'])

        return df

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df
    
    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = YahooDownloader(start_date = df.date.min(),
                                end_date = df.date.max(),
                                ticker_list = ["^VIX"]).fetch_data()
        vix = df_vix[['date','close']]
        vix.columns = ['date','vix']

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            #cov_temp = hist_price.cov()
            #current_temp=(current_price - np.mean(hist_price,axis=0))
            
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index




