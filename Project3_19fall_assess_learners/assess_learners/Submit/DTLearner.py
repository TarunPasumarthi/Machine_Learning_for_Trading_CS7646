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
                          
class DTLearner(object):                          
                          
    def __init__(self, leaf_size = 1, verbose = True):
        self.leaf_size=leaf_size 
                          
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
        self.tree = self.build_tree(newdataX)
    
    def build_tree(self, data):
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1,data[0][-1],-1,-1]])
        elif np.all(data[:,-1]==data[0,-1]):
            return np.array([[-1,data[0][-1],-1,-1]])
        else:
            corr_arr = []
            for i in range(data.shape[1]-1):
                col= data[:,i]
                corr=np.corrcoef(col,data[:,-1])[0,1]
                corr_arr.append(corr)
            corr_arr = np.absolute(corr_arr)
            best_i = np.argmax(corr_arr)
            splitVal= np.median(data[:,best_i],axis=0)
            
            if (splitVal== np.max(data[:,best_i], axis=0)):
                temp = np.argmax(data[:,best_i])
                return np.array([[-1,data[temp][-1],-1,-1]])

            lefttree=self.build_tree(data[data[:,best_i]<=splitVal])
            righttree=self.build_tree(data[data[:,best_i]>splitVal])
            root=np.array([[best_i,splitVal,1,lefttree.shape[0]+1]])

            return np.vstack((np.vstack((root,lefttree)),righttree))

                          
    def query(self,points):                          
        """                          
        @summary: Estimate a set of test points given the model we built.                          
        @param points: should be a numpy array with each row corresponding to a specific query.                          
        @returns the estimated values according to the saved model.                          
        """
        ret_vals=[]
        for point in points:
            result= self.query_tree(point,node=0)
            ret_vals.append(result)
        return np.array(ret_vals)
    
    def query_tree(self, point, node):
        index= int(self.tree[node,0])
        splitVal= self.tree[node,1]
        
        if(index==-1):
            return splitVal
        elif (point[index]<=splitVal):
            left= self.tree[node,2]
            new_node=int(node+left)
            return self.query_tree(point,new_node)
        else:
            right= self.tree[node,3]
            new_node=int(node+right)
            return self.query_tree(point,new_node)
        
    def num_leafs(self):
        leafs = 0
        for i in range(self.tree.shape[0]):
            if self.tree[i][0] == -1:
                leafs = leafs+1
        return leafs

                          
if __name__=="__main__":                          
    print("the secret clue is 'zzyzx'")                          
