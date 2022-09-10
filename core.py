#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:05:54 2021

@author: dylancalame
"""
import numpy as np
import random
import math
from scipy.stats import norm

class ResultsData():
    
    def __init__(self,num_trials:int=None):
        
        self.purkinje_fr = [None]*num_trials
        self.nuclear_fr = [None]*num_trials
        self.w_weights = [None]*num_trials
        self.v_weights = [None]*num_trials
        self.error = [None]*num_trials
        self.pCspk = [None]*num_trials
        self.pf_base = []
        self.in_base = []
        
class CerebellarModel():
    
    def __init__(self,time:int=None,
                 pf_on_length:int=None,
                 num_inputs:int=None,
                 num_mod_inputs:int=None,
                 input_type:str=None,
                 num_trials:int=None,
                 expt_type:str=None,
                 perturb_time:int=None,
                 perturb_mag:int=None,
                 perturb_onset:int=None,
                 perturb_offset:int=None,
                 perturb_length:int=None,
                 perturb_bpratio:float=0.75,
                 a:float = 0.0243,
                 tau:float = 19.49,
                 c:float =  0.00833,
                 alpha_pf:float = 0.95,
                 beta_Cspk:float = 0.15):
        
        # parameters
        self.a = a
        self.tau = tau
        self.c = 0.5/(abs(0.5-perturb_bpratio)*perturb_mag)
        self.alpha_pf = alpha_pf
        self.beta_Cspk = beta_Cspk
        
        self.t = time
        self.num_trials = num_trials
        self.PC_0 = -norm.pdf(np.arange(-self.t/2, self.t/2, 1),0,self.t/4)*10000
        self.CbN_0 = np.zeros(shape=(self.t))
        
        self.inputtype = input_type
        self.pf_on_length = pf_on_length 
        self.in_on_length = pf_on_length 
        self.num_PC_inputs = num_inputs
        self.num_nuclear_inputs = num_inputs
        self.num_mod_inputs = num_mod_inputs
        
        self.expt_type=expt_type
        self.perturb_onset=perturb_onset
        self.perturb_offset=perturb_offset
        self.perturb_mag=perturb_mag
        self.perturb_time=perturb_time
        self.perturb_length=perturb_length
        self.perturb_bpratio = perturb_bpratio
        
        self.w_0 = np.ones(shape = self.num_PC_inputs)
        self.w_weight = np.ones(shape = self.num_PC_inputs)
        self.v_weight = np.ones(shape = self.num_nuclear_inputs) # only change if you want to modify synaptic strength into nuclei
        self.random_pert_idx = np.zeros(shape = self.num_trials)
        
        self.results= ResultsData(num_trials)
        self.results.stim = np.zeros(shape = self.num_trials)
        self.trial = 0
        
        
    def input_setup(self):
        
        """Set up temporal basis set of balanced parallel fiber activtiy across movement."""
    
        # set up PFs
        PF = np.zeros(shape = [self.num_PC_inputs,self.t])
        correct = False
        
        for i in range(self.num_PC_inputs):
            
            pf_on = np.zeros(shape = (self.t+(self.pf_on_length*2))) 
            
            if (i % 2) == 0: # positive modulated
                
                if self.inputtype == 'gaussian':
                # make sure random time of pf_on is in right bound
                    while correct == False:
                        # gaussian temporal activation profile
                        index = int(np.random.normal(self.t/2, self.t/4, 1))
                        if index>=0 and index<=self.t:
                            correct = True
                        else: 
                            correct = False
                            
                    # extend bounds of pf_on so that activation can span movement ends, will clip below
                    pf_on[index+self.pf_on_length:index+(2*self.pf_on_length)] = 1
                
                # uniform distribution
                elif self.inputtype == 'uniform':
                    index = random.randrange(400)
                    pf_on[index+self.pf_on_length:index+(2*self.pf_on_length)] = 1
                    
                
                # multiple pf activations per trial
                elif self.inputtype == 'multiact':
                    index1 = random.randrange(400)
                    index2 = random.randrange(400)
                    pf_on[index1+self.pf_on_length:index1+(2*self.pf_on_length)] = 1
                    pf_on[index2+self.pf_on_length:index2+(2*self.pf_on_length)] = 1
    
                # assign input FR (also clip back to length of trials)
                PF[i,:] = abs(self.PC_0*pf_on[self.pf_on_length:self.pf_on_length+self.t])
    
            else: # balanced negative modulation
                if self.inputtype == 'gaussian':
                    pf_on[index+self.pf_on_length:index+2*self.pf_on_length] = 1
                
                # uniform distribution
                elif self.inputtype == 'uniform':
                    pf_on[index+self.pf_on_length:index+2*self.pf_on_length] = 1
                    
                elif self.inputtype == 'multiact':
                    pf_on[index1+self.pf_on_length:index1+2*self.pf_on_length] = 1
                    pf_on[index2+self.pf_on_length:index2+2*self.pf_on_length] = 1
    
                PF[i,:] = -(abs(self.PC_0*pf_on[self.pf_on_length:self.pf_on_length+self.t]))
                
            correct = False
            
        # set up nuclear inputs (balanced with PC inputs)
        IN = np.zeros(shape = [self.num_nuclear_inputs,self.t])
        
        for i in range(self.num_nuclear_inputs):
            
            in_on = np.zeros(shape = (self.t+(self.in_on_length*2)))
            
            if (i % 2) == 0:
                
                if self.inputtype == 'gaussian':
                # make sure random time of pf_on is in right bound
                    while correct == False:
                        index = int(np.random.normal(self.t/2, self.t/4, 1))
                        if index>=0 and index<=self.t:
                            correct = True
                        else: 
                            correct = False
                            
                    in_on[index+self.in_on_length:index+2*self.in_on_length] = 1
                
                # uniform distribution
                elif self.inputtype == 'uniform':
                    index = random.randrange(400)
                    in_on[index+self.in_on_length:index+2*self.in_on_length] = 1
                
                # multiple activations uniform
                elif self.inputtype == 'multiact':
                    index1 = random.randrange(400)
                    index2 = random.randrange(400)
                    in_on[index1+self.in_on_length:index1+2*self.in_on_length] = 1
                    in_on[index2+self.in_on_length:index2+2*self.in_on_length] = 1
                    
                # assign PF FR
                IN[i,:] = abs(self.PC_0*in_on[self.in_on_length:self.in_on_length+self.t])
                
            else: 
                if self.inputtype == 'gaussian':
                    in_on[index+self.in_on_length:index+2*self.in_on_length] = 1
                
                # uniform distribution
                elif self.inputtype == 'uniform':
                    in_on[index+self.in_on_length:index+2*self.in_on_length] = 1
    
                # multiple activations uniform
                elif self.inputtype == 'multiact':
                    in_on[index1+self.in_on_length:index1+2*self.in_on_length] = 1
                    in_on[index2+self.in_on_length:index2+2*self.in_on_length] = 1
                    
                IN[i,:] = -(abs(self.PC_0*in_on[self.in_on_length:self.in_on_length+self.t]))
                
            correct = False
                
        self.results.pf_base = PF
        self.results.in_base = IN
    
    def get_random_list(self):
        """         
        Select random parallel fibers to perturb. 
        Allows for repeated selection of a parallel fiber.
        """
        randomlist = np.zeros(self.num_PC_inputs)
        for i in range(int(self.num_mod_inputs*self.perturb_bpratio)):
            chosen_pf=False
            while chosen_pf is False:
                n = random.randint(0,self.num_PC_inputs-1)
                if n%2==0:
                    randomlist[n] = randomlist[n] + 1
                    chosen_pf =True
                    
        for i in range(int(self.num_mod_inputs*(1-self.perturb_bpratio))):
            chosen_pf=False
            while chosen_pf is False:
                n = random.randint(0,self.num_PC_inputs-1)
                if n%2==1:
                    randomlist[n] = randomlist[n] + 1
                    chosen_pf =True           
        
        self.randomlist = randomlist
        
    def perturbation_add(self):
        """ Get vector of perturbation magnitude if in perturbation block. """
        
        PF_pert = np.zeros(shape = (self.t))
            
        if self.trial >= self.perturb_onset and self.trial <= self.perturb_offset:
            
            self.results.stim[self.trial]=1
            
            if self.expt_type == "fixed_position":
                idx = self.perturb_time
            elif self.expt_type == "random_position":
                idx = random.randrange(0,self.t-self.perturb_length)
                self.random_pert_idx[self.trial] = idx
                
            PF_pert[idx:idx+self.perturb_length] = self.perturb_mag
            
        self.pf_pert = PF_pert
    
                
    def run_trial(self):
        """ Run trial based on with given parallel fiber weights with or without perturbation."""
        # add perturbation to synapses if necessary
        
        # add pert to random pfs
        PFplus = np.zeros(self.results.pf_base.shape)
        for i in range(self.num_PC_inputs):
            if self.randomlist[i] != 0: 
                # increase activity if PF, decrease if IN
                if i%2==0:
                    PFplus[i,:] = self.results.pf_base[i,:]+((self.pf_pert*self.randomlist[i])/self.num_mod_inputs) 
                else:
                    PFplus[i,:] = self.results.pf_base[i,:]-((self.pf_pert*self.randomlist[i])/self.num_mod_inputs) 
            else:
                PFplus[i,:] = self.results.pf_base[i,:]
                 
        # sum up PF synapses
        PF_net = sum([self.w_weight[i]*PFplus[i,:] for i in range(self.num_PC_inputs)])
        
        # get PC FR
        PC = PF_net+self.PC_0
        
        # sum up IN synapses
        IN_net = sum([self.v_weight[i]*self.results.in_base[i,:] for i in range(self.num_nuclear_inputs)])
        
        # get nuclear FR
        CbN = (IN_net-self.CbN_0)-(PC-self.PC_0)
     
        # get error
        Error = -self.c*CbN
        
        # save results
        self.results.purkinje_fr[self.trial] = PC
        self.results.nuclear_fr[self.trial] = CbN
        self.results.w_weights[self.trial] = self.w_weight
        self.results.v_weights[self.trial] = self.v_weight
        self.results.error[self.trial] = Error
        self.pf_plus = PFplus
        
        # log weighted PF activity at keypoints in expt
        if self.trial == self.perturb_onset-1:
            self.results.PF_endbaseline = np.array([self.w_weight[i]*PFplus[i,:] for i in range (len(self.w_weight))])
        elif self.trial == self.perturb_onset:
            self.results.PF_startperturbation = np.array([self.w_weight[i]*PFplus[i,:] for i in range (len(self.w_weight))])
        elif self.trial == self.perturb_offset+1:
            self.results.PF_startwashout = np.array([self.w_weight[i]*PFplus[i,:] for i in range (len(self.w_weight))])
            self.results.PF_weights_startwashout = self.w_weight


    def Cspk_prob(self):
        """ 
        Calculate probability of CS over time. 
        """
        Error = self.results.error[self.trial]
        a = self.a
        tau = self.tau
        pCspk = np.array([(a/(1 + math.exp(-tau * Error[t].mean())) - (a/2)) for t in range(len(Error))])
        self.results.pCspk[self.trial] = pCspk
        
    def update_pf_weights(self):
        """Update weights of parallel fibers based on error and pf activity"""
        
        # array for new weights
        w_new = np.zeros(shape = (self.num_PC_inputs))
        # update weights conditional on whether PF is firing over 0 
        for i in range(self.num_PC_inputs):
            arr = self.results.pCspk[self.trial]
            # find where activity was greater than 0 and select Cspk probability during this window (mimics LTD plasticity contingent on Ca++ buildup from pf activation)
            activity = arr[self.pf_plus[i,:]>0].mean()
            # if input was active use Cspk learning rule
            if not math.isnan(activity):
                w_new[i] = self.w_weight[i]-(1-self.alpha_pf)*(self.w_weight[i]-self.w_0[i])-(activity * self.beta_Cspk)
            else:
                w_new[i] = self.w_weight[i]-(1-self.alpha_pf)*(self.w_weight[i]-self.w_0[i])
                
        self.w_weight = w_new
        
    def run_experiment(self):
        
        self.input_setup()
        self.get_random_list()
        for _ in range(self.num_trials):
            self.perturbation_add()
            self.run_trial()
            self.Cspk_prob()
            self.update_pf_weights()
            self.trial += 1
            
        return self.results
    
    def updateweights_in(self):
        # array for new weights
        v_new = np.zeros(shape = (self.num_nuclear_inputs))
        for i in range(self.num_nuclear_inputs):           
            v_new[i] = self.v_weight[i]+self.eta*(self.PC_0-self.results.purkinje_fr[self.trial])*self.results.in_base[i]
        self.v_weight = v_new





