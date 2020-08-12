"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: tpasumarthi3 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903085537 (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import random as rand  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class QLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        self.verbose= verbose
        self.num_states= num_states		   	  			  	 		  		  		    	 		 		   		 		  
        self.num_actions= num_actions
        self.alpha= alpha
        self.gamma= gamma
        self.rar= rar
        self.radr= radr
        self.dyna= dyna 		   	  			  	 		  		  		    	 		 		   		 		  
        self.s= 0  		   	  			  	 		  		  		    	 		 		   		 		  
        self.a= 0
        self.Q= np.zeros((self.num_states, self.num_actions))
        self.s_arr,self.a_arr,self.sp_arr,self.r_arr=[],[],[],[]

    def author(self):
        return 'tpasumarthi3'  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        self.s = s
        if (rand.random() < self.rar):
            action= rand.randint(0, self.num_actions-1)
        else: 
            action= np.argmax(self.Q[s])

        if self.verbose: print(f"s = {s}, a = {action}")  		   	  			  	 		  		  		    	 		 		   		 		  
        return action  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def query(self,s_prime,r):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        self.Q[self.s,self.a]= ((1-self.alpha) * self.Q[self.s,self.a]) + (self.alpha * (r + (self.gamma * self.Q[s_prime, np.argmax( self.Q[s_prime] ) ] ) ) )

        self.s_arr.append(self.s)
        self.a_arr.append(self.a)
        self.sp_arr.append(s_prime)
        self.r_arr.append(r)
        rn= np.random.randint(len(self.s_arr), size=self.dyna)

        for i in range(self.dyna):
            idx= rn[i]
            ds= self.s_arr[idx]
            da= self.a_arr[idx]
            dsp= self.sp_arr[idx]
            dr= self.r_arr[idx]
            self.Q[ds,da]= ((1-self.alpha) * self.Q[ds,da]) + (self.alpha * (dr + (self.gamma * self.Q[dsp, np.argmax( self.Q[dsp] ) ] ) ) )

        if (rand.random() < self.rar):
            action= rand.randint(0, self.num_actions-1)
        else: 
            action= np.argmax(self.Q[s_prime])

        self.s= s_prime
        self.a= action
        self.rar= self.rar * self.radr

        if self.verbose: print(f"s = {s_prime}, a = {action}, r={r}")  		   	  			  	 		  		  		    	 		 		   		 		  
        return action  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		   	  			  	 		  		  		    	 		 		   		 		  
