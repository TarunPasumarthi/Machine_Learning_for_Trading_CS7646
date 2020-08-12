import datetime as dt
import pandas as pd
import util as ut
import random
import numpy as np
from util import *
from marketsimcode import *
import StrategyLearner as sl
from ManualStrategy import *
import matplotlib.pyplot as plt
#%matplotlib inline

def author(self):                          
        return 'tpasumarthi3' # replace tb34 with your Georgia Tech usernam
    
def stats(port_vals):
    pv=port_vals.values
    cr= (pv[-1]/pv[0])-1
    adr= (pv[1:]/pv[:-1]).mean()-1
    stddr= (pv[1:]/pv[:-1]).std()
    sr= (np.sqrt(252.0)*adr)/stddr
    return cr,adr,stddr,sr
    
def e1():
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = ['JPM']
    prices_all = ut.get_data(symbol, pd.date_range(sd, ed))
    
    #benchmark
    bench= pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    bench.loc[0]= [prices_all.index[0],'JPM','BUY',1000]
    bench.loc[1]= [prices_all.index[-1],'JPM','SELL',1000]
    bench_port_val= compute_portvals(bench,100000,0,0)
    cr,adr,stddr,sr= stats(bench_port_val)
    print("Benchmark")
    print("Cumulative Retrun: "+str(cr))
    print("Average Daily Return: "+str(adr))
    print("Standard Deviation Daily Return: "+ str(stddr))
    print("Sharpe Ratio: "+str(sr))
    print()
    
    #manual
    manual= testPolicy(symbol=['JPM'],sd=sd, ed=ed, sv=100000)
    manual_port_val= compute_portvals(manual,100000,0,0)
    cr,adr,stddr,sr= stats(manual_port_val)
    print("Manual")
    print("Cumulative Retrun: "+str(cr))
    print("Average Daily Return: "+str(adr))
    print("Standard Deviation Daily Return: "+ str(stddr))
    print("Sharpe Ratio: "+str(sr))
    print()
    
    #learner
    learner= sl.StrategyLearner(impact=0.0)
    learner.addEvidence(symbol="JPM", sd=sd, ed=ed, sv=100000)
    res= learner.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=100000)
    trades= pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    prices=res.values.T[0]
    for i in range(res.shape[0]):
        if(prices[i] == 1000):
            trades.loc[i]= [res.index[i],'JPM','BUY',1000]
        elif(prices[i] == -1000):
            trades.loc[i]= [res.index[i],'JPM','SELL',1000]
        elif(prices[i] == 2000):
            trades.loc[i]= [res.index[i],'JPM','BUY',2000]
        elif(prices[i] == -2000):
            trades.loc[i]= [res.index[i],'JPM','SELL',2000]
    learner_port_val = compute_portvals(trades,100000,0,0)
    cr,adr,stddr,sr= stats(learner_port_val)
    print("Learner")
    print("Cumulative Retrun: "+str(cr))
    print("Average Daily Return: "+str(adr))
    print("Standard Deviation Daily Return: "+ str(stddr))
    print("Sharpe Ratio: "+str(sr))

    #graphing
    bench_port_val= bench_port_val/bench_port_val[0]
    manual_port_val= manual_port_val/manual_port_val[0]
    learner_port_val= learner_port_val/learner_port_val[0]
    bench_port_val.plot(color="black", label='Benchmark')
    manual_port_val.plot(color="orange", label="Manual Strategy")
    learner_port_val.plot(color="red", label='Strategy Learner')
    plt.title("Experiment 1")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.savefig('e1.png')
    

if __name__=="__main__":
    e1()
    
