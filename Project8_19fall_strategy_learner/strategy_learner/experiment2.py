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

def to_comp(prices):
    trades= pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    p=prices.values.T[0]
    for i in range(prices.shape[0]):
        if(p[i] == 1000):
            trades.loc[i]= [prices.index[i],'JPM','BUY',1000]
        elif(p[i] == -1000):
            trades.loc[i]= [prices.index[i],'JPM','SELL',1000]
        elif(p[i] == 2000):
            trades.loc[i]= [prices.index[i],'JPM','BUY',2000]
        elif(p[i] == -2000):
            trades.loc[i]= [prices.index[i],'JPM','SELL',2000]
    return trades

def e2():
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    symbol = ['JPM']
    prices_all = ut.get_data(symbol, pd.date_range(sd, ed))
    impacts=[0.05,0.005,0.0005]
    pvs=[]

    for im in impacts:
        learner= sl.StrategyLearner(impact=im)
        learner.addEvidence(symbol="JPM", sd=sd, ed=ed, sv=100000)
        res= learner.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=100000)
        learner_port_val = compute_portvals(to_comp(res),100000,0,im)
        pvs.append(learner_port_val)
        cr,adr,stddr,sr= stats(learner_port_val)
        print("Learner_impact: "+str(im))
        print("Cumulative Retrun: "+str(cr))
        print("Average Daily Return: "+str(adr))
        print("Standard Deviation Daily Return: "+ str(stddr))
        print("Sharpe Ratio: "+str(sr))
        print()

    p1= pvs[0]/pvs[0][0]
    p2= pvs[1]/pvs[1][0]
    p3= pvs[2]/pvs[2][0]
    p1.plot(color="red", label='impact = 0.05')
    p2.plot(color="blue", label="impact = 0.005")
    p3.plot(color="purple", label='impact = 0.0005')
    plt.title("Experiment 2")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.savefig('e2.png')

if __name__=="__main__":
    e2()
    