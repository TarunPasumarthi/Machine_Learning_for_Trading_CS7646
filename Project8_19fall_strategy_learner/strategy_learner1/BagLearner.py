"""                          
A simple wrapper for linear regression.  (c) 2015 Tucker Balch                          
                          
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
import RTLearner as rl
import random
                          
class BagLearner(object):                          
                          
    def __init__(self, learner = rl.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):
        self.bags= bags
        self.boost= boost
        self.verbose=verbose
        self.learners=[]
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))
                          
    def author(self):                          
        return 'tpasumarthi3' # replace tb34 with your Georgia Tech username                          
                          
    def addEvidence(self,dataX,dataY):                          
        """                          
        @summary: Add training data to learner                          
        @param dataX: X values of data to add                          
        @param dataY: the Y training values                          
        """                          
                          
        # slap on 1s column so linear regression finds a constant term                          
        newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])                          
        newdataX[:,0:dataX.shape[1]]=dataX
        newdataX[:,-1]=dataY
        
        for i in range(self.bags):
            rows= dataX.shape[0]
            idx= np.random.choice(rows,rows,replace=True)
            temp_dataX= newdataX[idx,:-1]
            temp_dataY= newdataX[idx,-1]
            self.learners[i].addEvidence(temp_dataX,temp_dataY)
                          
    def query(self,points):                          
        """                          
        @summary: Estimate a set of test points given the model we built.                          
        @param points: should be a numpy array with each row corresponding to a specific query.                          
        @returns the estimated values according to the saved model.                          
        """
        ret_vals= np.zeros([self.bags, points.shape[0]])
        for i in range (self.bags):
            ret_vals[i]= self.learners[i].query(points)
        mean= np.mean(ret_vals,axis=0)
        return mean
                          
if __name__=="__main__":                          
    print("the secret clue is 'zzyzx'")                          
