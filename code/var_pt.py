import numpy as np
import warnings
warnings.filterwarnings("ignore")

import nrpt.py

def concatenate_paths(schedule_target_to_var, schedule_fixed_to_target, log_target, log_fixed_ref, current_var_distribution):
    path_target_to_var = path(schedule_target_to_var, log_target, current_ref_family_member)
    path_fixed_to_target = path(schedule_fixed_to_target, log_fixed_ref, log_target)
    
    return path_fixed_to_target.append(path_target_to_var.copy())
    
def concatenate_schedules(schedule_target_to_var, schedule_fixed_to_target):
    schedule1 = 0.5 * schedule_target_to_var
    schedule2 = 1 - 0.5 * schedule_fixed_to_target
    
    return schedule2.append(schedule1.copy())


def update_schedule(reject_rates, schedule):
    return schedule

def update_reference(samples):
    return 1
    
## Stabilized variational PT implementation!
def variational_PT(initial_state, num_chains, num_tuning_rounds, log_target, log_var_family, initial_phi, log_fixed_ref):
    schedule_target_to_var, schedule_fixed_to_target = np.linspace(0, 1, (num_chains / 2) + 1)
    curr_state = initial_state
    curr_phi = initial_phi
    
    for r in range(num_tuning_rounds):
        T = 2**r
        
        full_path = concatenate_paths(schedule_target_to_var, schedule_fixed_to_target, log_target, log_fixed_ref, log_var_family(curr_phi))
        full_schedule = concatenate_schedules(schedule_target_to_var, schedule_fixed_to_target)
        
        result = vanilla_NRPT(initial_state, full_schedule, T, full_path)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        schedule_target_to_var = update_schedule(reject_rates, schedule_target_to_var)
        schedule_fixed_to_target = update_schedule(reject_rates, schedule_fixed_to_target)
        
        curr_state = samples
        curr_phi = update_reference(samples)
    
    return curr_state
        
        
        
        
        
        
        
        
        
        
    