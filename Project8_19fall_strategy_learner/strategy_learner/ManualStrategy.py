import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import *
from marketsimcode import *
from indicators import *

def author(self):                          
        return 'tpasumarthi3' # replace tb34 with your Georgia Tech usernam

def testPolicy(symbol,sd,ed,sv):
    dates = pd.date_range(sd, ed)
    prices_all = get_data(symbol, dates)
    prices = prices_all[symbol]
    
    sma,pdsma= get_SMA(prices,symbol)
    pdsma= pdsma.T.iloc[0]
    upper, lower= get_BB(prices,symbol)
    momentum= get_Momentum(prices, symbol)
    
    shares=0
    s=symbol[0]
    trades = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])

    for i in range(prices.shape[0]- 1):
        if(shares == 1000 and (pdsma.iloc[i]> 1.4 or upper.iloc[i]> 0.8 or momentum.iloc[i]> 0.2)):
            trades.loc[i] = [prices.index[i],s,'SELL',2000]
            shares= -1000
        elif(shares == 1000 and (pdsma.iloc[i]> 1.0 or upper.iloc[i]> 0.7 or momentum.iloc[i]> 0.1)):
            trades.loc[i] = [prices.index[i],s,'SELL',1000]
            shares= 0
        elif(shares == -1000 and (pdsma.iloc[i]< 0.7 or upper.iloc[i]< 0.2 or momentum.iloc[i]< -0.2)):
            trades.loc[i] = [prices.index[i],s,'BUY',2000]
            shares= 1000
        elif(shares == -1000 and (pdsma.iloc[i]< 0.6 or upper.iloc[i]< 0.3 or momentum.iloc[i]< -0.1)):
            trades.loc[i] = [prices.index[i],s,'BUY',1000]
            shares= 0
        elif(shares == 0 and (pdsma.iloc[i]< 0.6  or upper.iloc[i]< 0.3 or momentum.iloc[i]< -0.1)):
            trades.loc[i] = [prices.index[i],s,'BUY',1000]
            shares= 1000
        elif(shares == 0 and (pdsma.iloc[i]> 1.0 or upper.iloc[i]> 0.7 or momentum.iloc[i]> 0.1)):
            trades.loc[i] = [prices.index[i],s,'SELL',1000]
            shares= -1000

    if(shares==1000):
        trades.loc[prices.shape[0]] = [prices.index[i],s,'SELL',1000]
    if(shares == -1000):
        trades.loc[prices.shape[0]] = [prices.index[i],s,'BUY',1000]


    return trades
