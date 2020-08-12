import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import math

from util import get_data
from marketsimcode import compute_portvals

#%matplotlib inline


def testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    df= get_data([symbol],pd.date_range(sd, ed))
    df= df.fillna(method= "ffill").fillna(method= "bfill")

    first= True
    last_buy= False
    date= []
    symb= []
    order= []
    shares= []

    for i in range(df.shape[0]-1):
        curr= df[symbol][i]
        aft= df[symbol][i+1]
        if (first and curr< aft):
            date.append(df.index[i])
            symb.append(symbol)
            order.append("BUY")
            shares.append(1000)
            first=False
            last_buy=True
        elif (first and curr > aft):
            date.append(df.index[i])
            symb.append(symbol)
            order.append("SELL")
            shares.append(1000)
            first=False
            last_buy=False
        elif (curr < aft and  not last_buy):
            date.append(df.index[i])
            symb.append(symbol)
            order.append("BUY")
            shares.append(2000)
            last_buy=True
        elif (curr > aft and last_buy):
            date.append(df.index[i])
            symb.append(symbol)
            order.append("SELL")
            shares.append(2000)
            last_buy=False
            
    if(last_buy):       
        date.append(df.index[-1])
        symb.append(symbol)
        order.append("SELL")
        shares.append(1000)
    else:
        date.append(df.index[-1])
        symb.append(symbol)
        order.append("BUY")
        shares.append(1000)

    df_trades = pd.DataFrame(data=date, index=np.arange(len(date)))
    df_trades["Symbol"]= symb
    df_trades["Order"]= order
    df_trades["Shares"]= shares
    df_trades.columns= ["Date","Symbol","Order","Shares"]
    
    return df_trades



def benchMark(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    df = get_data([symbol],pd.date_range(sd, ed))
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    date = [df.index[0], df.index[-1]]
    symb= [symbol, symbol]
    order =["BUY", "SELL"]
    share = [1000, 1000]
    
    df_trades = pd.DataFrame(data= date, index= range(len(date)))
    df_trades["Symbol"]= symb
    df_trades["Order"]= order
    df_trades["Shares"]= share
    df_trades.columns= ["Date", "Symbol", "Order", "Shares"]
    
    return df_trades



def stats():
    sd = "2008-1-1"
    ed = "2009-12-31"
    symbol = "JPM"

    df_trades= testPolicy(symbol, sd, ed, 100000)
    port_vals= compute_portvals(df_trades, start_val = 100000, commission=0.00, impact=0.00)
    port_vals_norm= port_vals/port_vals.iloc[0]
    daily_ret= (port_vals/port_vals.shift(1))- 1
    cum_ret= (port_vals[-1]/port_vals[0])-1
    avg_daily_ret= daily_ret.mean()
    std_daily_ret= daily_ret.std()
    sharpe_ratio= math.sqrt(252.0)*(avg_daily_ret/std_daily_ret)
    

    df_b= benchMark(symbol, sd, ed, 100000)
    port_vals_b= compute_portvals(df_b, start_val = 100000, commission=0.00, impact=0.00)
    port_vals_b_norm= port_vals_b/port_vals_b.iloc[0]
    daily_ret_b= (port_vals_b/port_vals_b.shift(1))- 1
    cum_ret_bchm= (port_vals_b[-1]/port_vals_b[0])-1
    avg_daily_ret_b= daily_ret_b.mean()
    std_daily_ret_b= daily_ret_b.std()
    sharpe_ratio_b= math.sqrt(252.0)*(avg_daily_ret_b/std_daily_ret_b)

    print("Best Policy")
    print(f"Cumulative Return: {cum_ret}")
    print(f"Average Daily Return: {avg_daily_ret}")
    print(f"Standard Deviation: {std_daily_ret}")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Final Value: {port_vals[-1]}")
    print()
    
    print("Benchmark")
    print(f"Cumulative Return: {cum_ret_bchm}")
    print(f"Average Daily Return: {avg_daily_ret_b}")
    print(f"Standard Deviation: {std_daily_ret_b}")
    print(f"Sharpe Ratio: {sharpe_ratio_b}")
    print(f"Final Value: {port_vals_b[-1]}")

    df_plot=port_vals_norm.to_frame().join(port_vals_b_norm.to_frame(),lsuffix= '', rsuffix= 'p')
    df_plot.columns= ["Best Policy","Benchmark"]
    graph = df_plot.plot(title="Value of Benchmark and Best Policy",color = ["red","green"])
    graph.set_xlabel("Date")
    graph.set_ylabel("Normalized Value")
    plt.savefig('Part2.png')
    
    

if __name__ == "__main__":
    stats()
    