
from core import CerebellarModel
from plotting import ModelPlotting

""" Example fixed-position stimulation experiment simulation""" 

# setup model variables
time=400 # trial time
pf_on_length=15 # length of time individual pfs are activated
num_inputs=2000 # number of of inputs
num_mod_inputs=200 # number of modified inputs
input_type='gaussian' # type of net input activity over time
num_trials=60 # number of trials
expt_type='fixed_position' # fixed_position or random_position
perturb_time=200 # time of perturbation onset
perturb_length=50 # length of perturbation
perturb_mag=80 # spikes added
perturb_onset=21 # trial where perturbation starts
perturb_offset=40 # trial where perturbation stops
perturb_bpratio=0.25 # ratio of parallel fibers to mlis activated

# run model
tcm = CerebellarModel(time=time,
                      pf_on_length=pf_on_length,
                      num_inputs=num_inputs,
                      num_mod_inputs=num_mod_inputs,
                      input_type=input_type,
                      num_trials=num_trials,
                      expt_type=expt_type,
                      perturb_time=perturb_time,
                      perturb_length=perturb_length,
                      perturb_mag=perturb_mag,
                      perturb_onset=perturb_onset,
                      perturb_offset=perturb_offset,
                      perturb_bpratio = perturb_bpratio)
results = tcm.run_experiment()

# plot results
mp = ModelPlotting(results)
mp = ModelPlotting(results)
mp.plot_trials(plot_washout=False)
mp.plot_trials(plot_stim=False)

PF_wash1 = mp.ordered_pf_activity(results.PF_endbaseline, results.PF_startwashout)
PF_base = mp.ordered_pf_activity(results.PF_endbaseline, results.PF_endbaseline)
pf_stim1 = mp.ordered_pf_activity(results.PF_endbaseline, results.PF_startperturbation)

mp.plot_temporal_inputs(pf_stim1)   
mp.plot_weight_change(PF_wash1,PF_base)

mp.plot_Error_Cspk_relationship(tcm)
ww_savg, ww_nsavg = mp.ordered_weights(results.PF_weights_startwashout, results.PF_endbaseline, tcm.randomlist)

#%% 
""" Example random-position stimulation experiment simulation""" 

# setup model variables
time=400 # trial time
pf_on_length=15 # length of time individual pfs are activated
num_inputs=2000 # number of of inputs
num_mod_inputs=200 # number of modified inputs
input_type='gaussian' # type of net input activity over time
num_trials=60 # number of trials
expt_type='random_position' # fixed_position or random_position
perturb_time=200 # time of perturbation onset
perturb_length=50 # length of perturbation
perturb_mag=80 # spikes added
perturb_onset=21 # trial where perturbation starts
perturb_offset=40 # trial where perturbation stops
perturb_bpratio=0.75 # ratio of parallel fibers to mlis activated, must be greater than 0.5 for learning

# run model
tcm = CerebellarModel(time=time,
                      pf_on_length=pf_on_length,
                      num_inputs=num_inputs,
                      num_mod_inputs=num_mod_inputs,
                      input_type=input_type,
                      num_trials=num_trials,
                      expt_type=expt_type,
                      perturb_time=perturb_time,
                      perturb_length=perturb_length,
                      perturb_mag=perturb_mag,
                      perturb_onset=perturb_onset,
                      perturb_offset=perturb_offset,
                      perturb_bpratio = perturb_bpratio)
results = tcm.run_experiment()


# plot results
mp = ModelPlotting(results)
mp.plot_trials(plot_washout=False)
mp.plot_trials(plot_stim=False)

PF_wash1 = mp.ordered_pf_activity(results.PF_endbaseline, results.PF_startwashout)
PF_base = mp.ordered_pf_activity(results.PF_endbaseline, results.PF_endbaseline)
pf_stim1 = mp.ordered_pf_activity(results.PF_endbaseline, results.PF_startperturbation)

mp.plot_temporal_inputs(pf_stim1)   
mp.plot_weight_change(PF_wash1,PF_base)

ww_savg, ww_nsavg = mp.ordered_weights(results.PF_weights_startwashout, results.PF_endbaseline, tcm.randomlist)



