import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator

import nrpt

def concatenate_paths(schedule_target_to_var, schedule_fixed_to_target, log_target, log_fixed_ref, current_var_distribution):
    path_target_to_var = nrpt.path(schedule_target_to_var, log_target, current_var_distribution)
    path_fixed_to_target = nrpt.path(schedule_fixed_to_target, log_fixed_ref, log_target)
    
    return path_fixed_to_target + path_target_to_var.copy()
    
def concatenate_schedules(schedule_target_to_var, schedule_fixed_to_target):
    schedule1 = 0.5 * schedule_target_to_var
    schedule2 = 1 - 0.5 * schedule_fixed_to_target
    
    return np.concatenate((schedule2, schedule1.copy()))


def bisect(a, b, func, epsilon):
    if b-a >= epsilon:
        c = (a+b)/2
        if func(c) == 0.0:
            return c
        elif func(a)*func(c) < 0:
            b = c
        else:
            a = c
        return bisect(a,b,func,epsilon)
    else:
        return (a+b)/2


def interpolate(reject_rates, schedule):
    length = len(schedule)
    lambda_hat = np.zeros(length)
    
    for i in range(length):
        lambda_hat[i] += sum(reject_rates[j] for j in range(i))
    
    return PchipInterpolator(schedule,lambda_hat)
    

def update_schedule(reject_rates, schedule):
    length = len(schedule)
    interpolation = interpolate(reject_rates, schedule)
    lambda1 = interpolation(1)
    
    new_schedule = np.zeros(length)
    for p in range(length):
        func = lambda x: interpolation(x) - (lambda1 * p) / length
        new_schedule[p] = bisect(0,1,func,epsilon=0.0001)
    
    return new_schedule
    
    
def update_reference(samples):
    return np.var(samples)
    
## Stabilized variational PT implementation!
def variational_PT(initial_state, num_chains, num_tuning_rounds, log_target, log_var_family, initial_phi, log_fixed_ref, gradients):
    schedule_target_to_var = np.linspace(0, 1, num_chains)
    schedule_fixed_to_target = np.linspace(0, 1, num_chains)
    curr_state = initial_state
    curr_phi = initial_phi
    
    for r in range(num_tuning_rounds):
        T = 2**r
        
        full_path = concatenate_paths(schedule_target_to_var, schedule_fixed_to_target, log_target, log_fixed_ref, log_var_family(curr_phi))
        full_schedule = concatenate_schedules(schedule_target_to_var, schedule_fixed_to_target)
        
        result = nrpt.vanilla_NRPT(initial_state, full_schedule, full_path, T, gradients)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        schedule_target_to_var = update_schedule(reject_rates, schedule_target_to_var)
        schedule_fixed_to_target = update_schedule(reject_rates, schedule_fixed_to_target)
        
        curr_state = samples
        curr_phi = update_reference(samples)
    
    return curr_state



## Toy example
log_fixed_reference = lambda x: -2 * x**2     ## N(0,0.25) reference
log_target = lambda x: -x**2 / 2    ## N(0,1) target

num_chains = 4
initial_state = [0.1] * (2*num_chains)
num_tuning_rounds = 7
initial_phi = 0.5

gradients = [
    lambda x: -4 * x,
    lambda x: -2.5 * x,
    lambda x: (-7/4) * x,
    lambda x: -x,
    lambda x: -0.4 * x,     ## PLACEHOLDER
    lambda x: -0.3 * x,     ## PLACEHOLDER
    lambda x: -0.2 * x,     ## PLACEHOLDER
    lambda x: -0.1 * x      ## PLACEHOLDER
]

def log_var_family(phi):
    return lambda x: -0.5 * (x**2 / phi)

def log_var_family_gradient(phi):
    return lambda x: -x / phi


result = variational_PT(initial_state, num_chains, num_tuning_rounds, log_target, log_var_family, initial_phi, log_fixed_reference, gradients)
print("The mean is:", np.mean(result))
print("The var is:", np.var(result))
        
        

        
        
        
        
        
    