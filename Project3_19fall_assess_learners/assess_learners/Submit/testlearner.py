"""                          
Test a learner.  (c) 2015 Tucker Balch                          
                          
Copyright 2018, Georgia Institute of Technology (Georgia Tech)                          
Atlanta, Georgia 30332                          
All Rights Reserved                          
                          
Template code for CS 4646/7646                          
                          
Georgia Tech asserts copyright ownership of this template and all derivative                          
works, including solutions to the projects assigned in this course. Students                          
and other users of this template code are advised not to share it with others                          
or to make it available on publicly viewable websites including repositories                          
such as github and gitlab.  This copyright statement should not be removed                          
or edited.                          
                          
We do grant permission to share solutions privately with non-students such                          
as potential employers. However, sharing with other current or future                          
students of CS 7646 is prohibited and subject to being investigated as a                          
GT honor code violation.                          
                          
-----do not edit anything above this line---                          
"""                          
                          
import numpy as np                          
import math                          
import LinRegLearner as lrl                          
import sys
import DTLearner as dl
import RTLearner as rl
import BagLearner as bl
import InsaneLearner as il
import matplotlib.pyplot as plt
import matplotlib
import time
                          
if __name__=="__main__":                          
    if len(sys.argv) != 2:                          
        print("Usage: python testlearner.py <filename>")                          
        sys.exit(1)  
    if sys.argv[1] == "Data/Istanbul.csv":
        inf = open(sys.argv[1])
        data = np.genfromtxt(inf,delimiter=',')
        data = data[1:,1:]
    else:
        inf = open(sys.argv[1])
        data = np.array(list([map(float,s.strip().split(',')) for s in inf.readlines()]))                         
                          
    # compute how much of the data is training and testing                          
    train_rows = int(0.3* data.shape[0])                          
    test_rows = data.shape[0] - train_rows                          
                          
    # separate out training and testing data                          
    trainX = data[:train_rows,0:-1]                          
    trainY = data[:train_rows,-1]                          
    testX = data[train_rows:,0:-1]                          
    testY = data[train_rows:,-1]
    
    ###question 1
    rmse_arr=[]
    rmse_arr2=[]
    for i in range(1,31):
        learner = dl.DTLearner(leaf_size=i,verbose = True)
        learner.addEvidence(trainX, trainY)
        pred=learner.query(trainX)
        rmse = np.sqrt(np.mean((pred-trainY)**2))
        rmse_arr.append(rmse)
        pred2=learner.query(testX)
        rmse2=np.sqrt(np.mean((pred2-testY)**2))
        rmse_arr2.append(rmse2)
        
    plt.figure(1)
    plt.plot(rmse_arr)
    plt.plot(rmse_arr2)
    plt.xlim(30,0)
    plt.ylim(0,0.015)
    plt.title("RMSE vs Leaf Size: DTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend(["In-Sample", "Out-Sample"])
    plt.savefig("q1.png")
    
    # compute how much of the data is training and testing                          
    train_rows = int(0.6* data.shape[0])                          
    test_rows = data.shape[0] - train_rows                          
                          
    # separate out training and testing data                          
    trainX = data[:train_rows,0:-1]                          
    trainY = data[:train_rows,-1]                          
    testX = data[train_rows:,0:-1]                          
    testY = data[train_rows:,-1]
    
    
    ###question2
    rmse_arr3=[]
    rmse_arr4=[]
    for i in range(1,31):
        learner=bl.BagLearner(learner=dl.DTLearner, kwargs={"leaf_size":i}, bags = 15, boost = False, verbose = False)
        learner.addEvidence(trainX, trainY)
        pred=learner.query(trainX)
        rmse = np.sqrt(np.mean((pred-trainY)**2))
        rmse_arr3.append(rmse)
        pred2=learner.query(testX)
        rmse2=np.sqrt(np.mean((pred2-testY)**2))
        rmse_arr4.append(rmse2)
        
    plt.figure(2)
    plt.plot(rmse_arr3)
    plt.plot(rmse_arr4)
    plt.xlim(30,0)
    plt.ylim(0,0.015)
    plt.title("RMSE vs Leaf Size: Bag Learner")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend(["In-Sample", "Out-Sample"])
    plt.savefig("q2.png")
    
    
    ###question 3: average tree depth
    dl_depth=[]
    rl_depth=[]
    for i in range(0, 31):
        learner = dl.DTLearner(leaf_size=i, verbose=True)
        learner.addEvidence(trainX, trainY)
        depth=int(np.log2(learner.num_leafs()))
        dl_depth.append(depth)

    for i in range(1, 31):
        learner = rl.RTLearner(leaf_size=i, verbose=True)
        learner.addEvidence(trainX, trainY)
        depth=int(np.log2(learner.num_leafs()))
        rl_depth.append(depth)
        
    plt.figure(3)
    plt.plot(dl_depth)
    plt.plot(rl_depth)
    plt.xlim(30,0)
    plt.ylim(0,10)
    plt.title("Average Tree Depth")
    plt.xlabel("Leaf Size")
    plt.ylabel("Level")
    plt.legend(["DTLEarner", "RTLearner"])
    plt.savefig("q3i.png")

    
    ###question 3: build time
    time_dl=[]
    time_rl=[]
    for i in range(1,31):
        start = time.time()
        learner = dl.DTLearner(leaf_size=i, verbose=True)
        learner.addEvidence(trainX, trainY)
        end = time.time()
        time_dl.append(end-start)
        
    for i in range(1,31):
        start = time.time()
        learner = rl.RTLearner(leaf_size=i, verbose=True)
        learner.addEvidence(trainX, trainY)
        end = time.time()
        time_rl.append(end-start)
    
    plt.figure(4)
    plt.plot(time_dl)
    plt.plot(time_rl)
    plt.xlim(30,0)
    plt.ylim(0,0.3)
    plt.title("Time to Build Tree: DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.ylabel("Time (s)")
    plt.legend(["DTLearner", "RTLearner"])
    plt.savefig("q3ii.png")                      
