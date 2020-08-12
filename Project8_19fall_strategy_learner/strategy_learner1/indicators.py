import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
#%matplotlib inline

def author():
    return 'tpasumarthi3'

def get_indicators(start, end, symbol, window=10):
    df= get_data([symbol],pd.date_range(start,end))
    df= df.fillna(method= 'ffill').fillna(method= 'bfill')
    df= df/df.iloc[0]
    df.columns = ["Price"]

    rm= df["Price"].rolling(window).mean()
    rstd= df["Price"].rolling(window).std()
    pdsma= df.divide(rm, axis=0)
    upper= rm + rstd*2
    lower= rm - rstd*2
    momentum= df["Price"]/df["Price"].shift(window) - 1

    return rm, rstd, pdsma, upper, lower, momentum

if __name__ == "__main__":
    indicators()
    