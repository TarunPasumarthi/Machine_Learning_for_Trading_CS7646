import pandas as pd 
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data 

def author():
        return 'tpasumarthi3' # replace tb34 with your Georgia Tech username.
    
def compute_portvals(orders, start_val = 1000000, commission=9.95, impact=0.005):
    #orders= pd.read_csv(orders_file)
    
    orders_array= orders.values
    start_date= orders_array[0,0]
    end_date= orders_array[-1,0]
    symbols= list(set(orders["Symbol"]))
    
    p_data= get_data(symbols,pd.date_range(start_date, end_date))
    t_data= get_data(symbols,pd.date_range(start_date, end_date))
    h_data= get_data(symbols,pd.date_range(start_date, end_date))
    
    p_data["Cash"]= 1
    p_data= p_data.drop("SPY", axis=1)
    
    t_data[symbols]= 0.0
    t_data["Cash"]= 0.0
    
    h_data[symbols]= 0.0
    h_data["Cash"]= 0.0
    h_data.iloc[0,-1]= start_val
    
    for index, row in orders.iterrows():
        d=row["Date"]
        sy=row["Symbol"]
        s=row["Shares"]
        
        if row["Order"]=="SELL":
            t_data.loc[d,sy]= t_data.loc[d,sy]-s
            t_data.loc[d,"Cash"]= t_data.loc[d,"Cash"] + (p_data.loc[d,sy] * s * (1-impact)) - commission
        else:
            t_data.loc[d,sy]= t_data.loc[d,sy]+s
            t_data.loc[d,"Cash"]= t_data.loc[d,"Cash"] + (p_data.loc[d,sy] * s * (-1-impact)) - commission
            
    h_data.iloc[0,0:-1]= t_data.iloc[0,0:-1]
    h_data.iloc[0,-1]= h_data.iloc[0,-1] + t_data.iloc[0,-1]
    
    for i in range(1, h_data.shape[0]):
        h_data.iloc[i,0:-1]= t_data.iloc[i,0:-1] + h_data.iloc[i-1,0:-1]
        h_data.iloc[i,-1]= t_data.iloc[i,-1] + h_data.iloc[i-1,-1]
        
    port_vals= (p_data * h_data).sum(axis=1)
    return port_vals
