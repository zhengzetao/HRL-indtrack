"""Contains methods and classes to collect data from
Yahoo Finance API
"""
import os
import pandas as pd
import yfinance as yf


class DataLoader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data as dataframe

    """

    def __init__(self, portfolio_name: str, start_date: str, end_date: str):

        self.portfolio_name = portfolio_name
        self.start_date = start_date
        self.end_date = end_date
        # self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        if self.portfolio_name == 'SP500':
            portfolio_save_path = "./datasets" + '/' + self.portfolio_name
            index_file_path = portfolio_save_path + '/^GSPC.csv'
        else:
            portfolio_save_path = "./datasets/ORL-data/" + self.portfolio_name + '/'
            index_file_path = portfolio_save_path  + '1.csv'
        if os.path.exists(portfolio_save_path):
            # standard_file = pd.read_csv(portfolio_save_path + '/' + self.ticker_list[0] + '.csv',index_col=False)
            # standard_date = pd.to_datetime(standard_file['Date'], errors='coerce')
            ticker_list = os.listdir(portfolio_save_path)
            # ticker_list.remove('^GSPC.csv') if self.portfolio_name == 'SP500' else ticker_list.remove('1.csv')
            for tic in ticker_list:
                temp_df = pd.read_csv(portfolio_save_path + '/' + tic,index_col=False)
                temp_df["tic"] = tic.split('.')[0]
                temp_df.rename(columns = {'Unnamed: 0':'Week_ID'}, inplace = True)
                data_df = data_df.append(temp_df)
            indexing = pd.read_csv(index_file_path, index_col=False)
            # data_df = data_df.insert(loc=0, column='ind', value=indexing.values)
            indexing.rename(columns = {'Unnamed: 0':'Week_ID','Close':'Indexing'}, inplace = True)
            # data_df.append(indexing)
        else:
            print("CANNOT FOUND SUCH DATASET!")

        print("Shape of DataFrame: ", data_df.shape)

        data_df = data_df.reset_index(drop=True)
        data_df = data_df.sort_values(by=['Week_ID','tic']).reset_index(drop=True)
        
        return data_df, indexing

    def fetch_data_SP500(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        portfolio_save_path = "./datasets/" + self.portfolio_name 
        index_file_path = portfolio_save_path + '/0.csv'
        if os.path.exists(portfolio_save_path):
            ticker_list = os.listdir(portfolio_save_path)
            # print(ticker_list)
            # standard_file = pd.read_csv(portfolio_save_path + '/' + ticker_list[0],index_col=False)
            # standard_date = pd.to_datetime(standard_file['Date'], errors='coerce')
            ticker_list.remove('0.csv') 
            ticker_list.insert(0,"0.csv") # ^GSPC as the first item
            print(ticker_list[:15])
            for tic in ticker_list[:]:
                temp_df = pd.read_csv(portfolio_save_path + '/' + tic,index_col=False)
                temp_df["tic"] = tic.split('.')[0]
                assert len(temp_df) == 3231, "The line number of {} is not equal to 3170".format(tic)
                temp_df.rename(columns = {'Unnamed: 0':'Week_ID'}, inplace = True)
                # condition1 = temp_df['Date']>=self.start_date
                # condition2 = temp_df['Date']<=self.end_date
                # temp_df = temp_df[condition1 & condition2]
                # temp_df['Date'] = pd.to_datetime(temp_df['Date'], errors='coerce')
                # temp_df = pd.merge(standard_date, temp_df, how='left', on="Date")
                temp_df = temp_df.fillna(method='ffill').fillna(method="bfill")
                # temp_df.set_index('Week_ID',inplace=True)
                data_df = data_df.append(temp_df)
            indexing = pd.read_csv(index_file_path, index_col=False)
            # data_df = data_df.insert(loc=0, column='ind', value=indexing.values)
            indexing.rename(columns = {'Unnamed: 0':'Week_ID'}, inplace = True)
            # data_df.append(indexing)
        else:
            print("CANNOT FOUND SUCH DATASET!")

        # reset the index, we want to use numbers as index instead of dates

        # data_df = data_df.reset_index()
        # print(temp_df.head(5))
        # try:
        #     # convert the column names to standardized names
        #     data_df.columns = [
        #         "date",
        #         "open",
        #         "high",
        #         "low",
        #         "close",
        #         "adjcp",
        #         "volume",
        #         "tic",
        #     ]
        #     # use adjusted close price instead of close price
        #     data_df["close"] = data_df["adjcp"]
        #     # drop the adjusted close price column
        #     data_df = data_df.drop("adjcp", 1)
        # except NotImplementedError:
        #     print("the features are not supported currently")
        # create day of the week column (monday = 0)
        # data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        # data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        # data_df = data_df.dropna()
        # data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display indexing: ", indexing.head())

        data_df = data_df.reset_index(drop=True)
        data_df = data_df.sort_values(by=['Week_ID','tic']).reset_index(drop=True)
        
        return data_df, indexing

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
