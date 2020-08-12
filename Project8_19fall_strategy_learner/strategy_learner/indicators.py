import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
#%matplotlib inline

def author():
    return 'tpasumarthi3'

def get_df(prices, symbol):
    df= prices[symbol]
    df= df.fillna(method= 'ffill').fillna(method= 'bfill')
    df= df/df.iloc[0]
    df.columns = ["Price"]
    return df

def get_SMA(prices,symbol, window=10):
    df= get_df(prices, symbol)
    rm= df["Price"].rolling(window).mean()
    pdsma= df.divide(rm, axis=0)
    return rm, pdsma

def get_BB(prices, symbol, window=10):
    df= get_df(prices, symbol)
    rm= df["Price"].rolling(window).mean()
    rstd= df["Price"].rolling(window).std()
    upper= rm + rstd*2
    lower= rm - rstd*2
    return upper, lower

def get_Momentum(prices, symbol, window=10):
    df= get_df(prices, symbol)
    momentum= df["Price"]/df["Price"].shift(window) - 1
    return momentum

if __name__ == "__main__":
    indicators()