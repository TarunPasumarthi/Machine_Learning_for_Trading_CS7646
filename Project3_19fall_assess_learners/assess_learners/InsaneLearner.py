import numpy as np
import LinRegLearner as lrl
import BagLearner as bl                 
class InsaneLearner(object):                                              
    def __init__(self, leaf_size = 1, verbose = True):
        for i in range(20):
            self.learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost= False, verbose= False)                 
    def author(self):                          
        return 'tpasumarthi3' # replace tb34 with your Georgia Tech username                                               
    def addEvidence(self,dataX,dataY):                          
        for i in range(20):
            self.learner.addEvidence(dataX,dataY)                     
    def query(self,points):                          
        ret_vals= []
        for i in range(20):
            ret_vals.append(self.learner.query(points))
        return np.mean(ret_vals, axis=0)                           
if __name__=="__main__":                          
    print("the secret clue is 'zzyzx'")
