import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.colors as mcolors
from core import ResultsData

parameters = {'font.sans-serif':'Arial',
      'figure.titlesize': 10,
'axes.labelsize': 10,
  'axes.titlesize': 10,
  'ytick.labelsize': 10,
  'xtick.labelsize': 10}
plt.rcParams.update(parameters)

class ModelPlotting():
    
    
    def __init__(self,results:ResultsData=None):
      
        self.results = results
        self.num_PC_inputs = results.pf_base.shape[0]

    def ordered_pf_activity(self,PF_base,PF_trial):
        
        """ Order PC input by time of activation or type (PF or MLI)"""
        act_up= np.zeros(shape = (self.num_PC_inputs,1))
        act_down= np.zeros(shape = (self.num_PC_inputs,1))
        for i in range(self.num_PC_inputs):
            if PF_base[i,:].mean()>0:
                act_length = np.where(PF_base[i,:]>0)
                act_up[i] = round(np.median(act_length))
            elif PF_base[i,:].mean()<0:
                act_length = np.where(PF_base[i,:]<0)
                act_down[i] = round(np.median(act_length))

        orderedlist = np.concatenate([act_up,act_down],axis =1)  
        foo = pd.DataFrame(columns = ['OrderMax', 'OrderMin'], data = orderedlist)
        PF_activity = pd.DataFrame(PF_trial)
        PF_activity = pd.concat([PF_activity,foo],axis = 1)
        PF_activity = PF_activity.sort_values(['OrderMax','OrderMin'])
        PF_activity = PF_activity.drop(columns = ['OrderMax','OrderMin'])
        
        return PF_activity
    
    def plot_temporal_inputs(self,PF_activity):
        
        """Plot inputs organized in ordered_pf_activity."""
        
        # setup colormap
        color1 = np.array([0.969,0.580,0.113,1])
        color2 = np.array([0.224,0.710,0.290,1])
        color3 = np.array([1,1,1,1])
        color_spectrum = np.vstack((color1, color3, color2))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', color_spectrum)
        
        # plot
        fig = plt.figure(figsize = (3,2),dpi=300)  
        ax1 = fig.add_subplot(111)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        plot = ax1.pcolor(PF_activity, cmap = mymap,vmin=-0.001, vmax=0.001)
        cbar = fig.colorbar(plot,ax = ax1)
        cbar.set_label('Activity', rotation=270)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Input')
        fig.tight_layout()
        
    def plot_weight_change(self,PF_activity,PF_activity_base):
        
        """ Plot change of weights from some trial relative to baseline."""
        
        # caculate % PFweight change from 
        norm_PF_pert = ((PF_activity/PF_activity_base)*100)-100

        # plot
        cmap = 'RdPu_r'
        fig = plt.figure(figsize = (3,2),dpi=300)  
        ax1 = fig.add_subplot(111)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        plot = ax1.pcolor(norm_PF_pert,cmap = cmap,vmin=-2, vmax=2)
        cbar = fig.colorbar(plot,ax = ax1)
        cbar.set_label('Change in Weight (%)', rotation=270)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Input')
        fig.tight_layout()
        
    def plot_trials(self,plot_baseline=True,plot_washout=True,plot_stim=True):

        """ Plot PC activity trial over trial."""
        
        fig = plt.figure(figsize = (3,4),dpi=300)  
        fig.tight_layout()
        ax1 = fig.add_subplot(111)
        ax1.set_ylim([-100, 40])
        ax1.set_title('PC activity')
        ax1.set_ylabel('SS Firing Rate (Hz)')
        fig.tight_layout()

        cmap1 = cm.get_cmap('Blues',100)
        cmap2 = cm.get_cmap('Reds',100)
        a=1
        b=1
        for i in range(60):
            if i==20 and plot_baseline:
                ax1.plot(self.results.purkinje_fr[i],linewidth=0.5,color = 'black')
            elif i>20 and i <=40 and plot_stim:# or i==30 or i == 40:
                ax1.plot(self.results.purkinje_fr[i],linewidth=0.5,color = cmap1(0.5+0.01*a))
                a+=1
            elif i>40 and plot_washout:
                ax1.plot(self.results.purkinje_fr[i],linewidth=0.5,color = cmap2(0.2+0.1*b))
                b+=1
                
                
    def ordered_weights(self,weights,PF_base,randomlist):
        
        """ Calculates and plots average weight of stimulated parallel fibers or nonstimulated fibers across time of trial."""
        
        stimPFs = np.where(randomlist > 0)[0]
        weight_map = np.zeros(shape = (self.num_PC_inputs,400))

        stim_map= np.zeros(shape = (len(randomlist),1))
        for i in range(self.num_PC_inputs):
            weight_map[i,np.where(abs(PF_base[i,:])>0)[0]] = weights[i]
            if i in stimPFs and i%2 ==0:
                 stim_map[i] = 1

        ww = pd.DataFrame(weight_map)
        ww2 = ww.replace(0, np.nan, inplace=False)
        
        # select stim'd pfs
        toDel = np.zeros(shape = (self.num_PC_inputs,1))
        for i in range(self.num_PC_inputs):
            if i in stimPFs and i%2 ==0:
                toDel[i] = 1
        ww_nostim = ww2[toDel == 0]     
        ww_stim = ww2[toDel == 1]
        
        ww_nsavg = ww_nostim.mean(axis = 0, skipna = True)-1
        ww_savg = ww_stim.mean(axis = 0, skipna = True)-1
        
        fig = plt.figure(figsize = (3,4),dpi=300)  
        fig.tight_layout()

        ax1 = fig.add_subplot(111)
        ax1.set_ylim([-0.025, 0.025])
        ax1.set_title('PC weight change')
        ax1.set_ylabel('delta weight')

        fig.tight_layout()
        ax1.plot(ww_nsavg)
        ax1.plot(ww_savg)
        
        return ww_savg, ww_nsavg
    
    def plot_Error_Cspk_relationship(self,tcm):
        a = tcm.a
        tau = tcm.tau
        Error = np.arange(-0.5,0.5,0.01)
        pCspk = np.array([(a/(1 + math.exp(-tau * Error[t].mean())) - (a/2)) for t in range(len(Error))])
        plt.plot(Error,pCspk)
        