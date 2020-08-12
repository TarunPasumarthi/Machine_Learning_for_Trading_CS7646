import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
#%matplotlib inline

def author():
    return 'tpasumarthi3'

def indicators():
    df= get_data(["JPM"],pd.date_range("2008-1-1","2009-12-31"), False)
    df= df.fillna(method= 'ffill').fillna(method= 'bfill')
    df= df/df.iloc[0]
    df.columns = ["Price"]

    rm= df["Price"].rolling(30).mean()
    rstd= df["Price"].rolling(30).std()
    pdsma= df.divide(rm, axis=0)
    upper= rm + rstd*2
    lower= rm - rstd*2
    momentum= df["Price"]/df["Price"].shift(30) - 1

    df_sma= df.copy()
    df_sma["SMA"]= rm
    df_sma["Price/SMA"]= pdsma
    graph= df_sma.plot(title="Simple Moving Average")
    graph.set_xlabel("Date")
    graph.set_ylabel("Normalized Prices")
    plt.savefig('Part1.1.png')

    df_BB= df.copy()
    df_BB["SMA"]= rm
    df_BB["Upper Bands"]= upper
    df_BB["Lower Bands"]= lower
    graph= df_BB.plot(title="Bollinger Bands")
    graph.set_xlabel("Date")
    graph.set_ylabel("Normalized Prices")
    plt.savefig('Part1.2.png')

    df_mom= df.copy()
    df_mom["Momentum"]= momentum
    graph= df_mom.plot(title="Normalized Price & Momentum")
    graph.set_xlabel("Date")
    graph.set_ylabel("Normalized Prices")
    plt.savefig('Part1.3.png')

if __name__ == "__main__":
    indicators()
    