import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator

import nrpt

sigma = 1

def RWMH_exploration_kernel(log_gamma, initial_x, num_iters):
    curr_point = initial_x
    samples = np.zeros(num_iters)
    
    for i in range(num_iters):
        new_point = np.random.normal(curr_point, 1, 1)
        
        p = np.min(1, gamma(new_point)/gamma(curr_point))
    
        if np.random.binomial(1, p):
            new_point = curr_point
    
        samples[i] = curr_point
        
    return samples


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
    return (np.mean(samples), np.var(samples))
    
## Stabilized variational PT implementation!
def variational_PT(initial_state, num_chains, num_tuning_rounds, log_target, log_var_family, initial_phi):
    schedule = np.linspace(0, 1, num_chains)
    curr_state = initial_state
    curr_phi = initial_phi
    
    for r in range(num_tuning_rounds):
        T = 2**r
        
        current_var_distribution = log_var_family(curr_phi[0], curr_phi[1])
        path = path(schedule, current_var_distribution, log_target)
        
        result = nrpt.vanilla_NRPT(initial_state, schedule, path, T, gradients)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        schedule_to_var_target = update_schedule(reject_rates, schedule_target_to_var)
        
        curr_state = samples
        curr_phi = update_reference(samples)
    
    return curr_state



## Toy example
def log_var_family(sigma, mu):
    return lambda x: -0.5 * ((x-mu) / sigma)**2

log_target = lambda x: -x**2 / 2    ## N(0,1) target

num_chains = 4
initial_state = [0.1] * (2*num_chains)
num_tuning_rounds = 3
initial_phi = (0,0.5)



samples = variational_PT(initial_state, num_chains, num_tuning_rounds, log_target, log_var_family, initial_phi, log_fixed_reference, gradients)
samples = [chain[num_chains-1] for chain in samples]
print("The mean is:", np.mean(samples))
print("The var is:", np.var(samples))
        
        

        
        
        
        
        
    