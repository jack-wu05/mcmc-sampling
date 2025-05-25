import numpy as np
import warnings
from scipy.stats import ks_2samp
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator

import nrpt
    
s = 1
    
def RWMH_exploration_kernel(log_gamma, initial_x, num_iters):
    curr_point = initial_x
    samples = np.zeros(num_iters)
    
    for i in range(num_iters):
        new_point = np.random.normal(curr_point, s, 1)
        
        p = min(1, np.exp(log_gamma(new_point)-log_gamma(curr_point)))
    
        if np.random.rand() < p:
            curr_point = new_point
    
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
    N = len(schedule)
    
    lambda_hat = np.zeros(N)
    for i in range(1,N):
        lambda_hat[i] = lambda_hat[i-1] + reject_rates[i-1]
    
    return PchipInterpolator(schedule,lambda_hat)
    

def update_schedule(reject_rates, schedule):
    N = len(schedule)
    
    interpolation = interpolate(reject_rates, schedule)
    lambda1 = interpolation(1)
    
    new_schedule = np.zeros(N)
    new_schedule[0] = 0.0
    new_schedule[-1] = 1.0
    
    for p in range(1,N-1):
        func = lambda x: interpolation(x) - (lambda1 * p) / (N-1)
        new_schedule[p] = bisect(0,1,func,epsilon=0.0001)**2

    return new_schedule

    
def update_reference(samples):
    return (np.mean(samples), np.std(samples))


def vanilla_NRPT_with_RWMH(initial_state, betas, log_annealing_path, num_iterations):
    num_distributions = len(betas)

    samples = []
    reject_rates = np.zeros(num_distributions-1)
    
    x_at_tminus1 = initial_state.copy()
    x_at_t = np.zeros(num_distributions)

    for t in range(num_iterations):
        for init in range(num_distributions):
            log_gamma = log_annealing_path[init]

            x_at_t[init] = RWMH_exploration_kernel(log_gamma, x_at_tminus1[init], 3)[-1]
        
        temp_alpha_vector = np.zeros(num_distributions)

        for i in range(num_distributions-1):
            temp_alpha_vector[i] = nrpt.alpha(log_annealing_path[i], log_annealing_path[i+1], x_at_t[i], x_at_t[i+1])
            reject_rates[i] = reject_rates[i] + (1 - temp_alpha_vector[i])/num_iterations
            
            if ((i%2==0 and t%2==0) or (i%2==1 and t%2==1)) and (np.random.rand() <= temp_alpha_vector[i]):
                x_at_t[i], x_at_t[i+1] = x_at_t[i+1], x_at_t[i]

        x_at_tminus1 = x_at_t.copy()
        samples.append(x_at_tminus1.copy())

    return {
        "reject_rates": reject_rates,
        "samples": samples
    }
    
    
## Variational PT implementation!
def variational_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target, log_var_family, initial_phi):
    schedule = np.linspace(0, 1, num_chains)
    curr_state = initial_state
    curr_phi = initial_phi
    
    toReturn = []
    
    for r in range(1, num_tuning_rounds):
        T = 2**r
        
        curr_var = log_var_family(curr_phi[0], curr_phi[1])
        path = nrpt.path(schedule, curr_var, log_target)
        
        result = vanilla_NRPT_with_RWMH(curr_state, schedule, path, T)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        if r >= 5: 
            for chain in samples: toReturn.append(chain[-1])
        
        schedule = update_schedule(reject_rates, schedule)
        curr_phi = update_reference([chain[-1] for chain in samples])
        curr_state = samples[-1]
    
    return toReturn


## Kolmogorov-Smirnov test for kernel pi-invariance using N(0,1)
# log_gamma = lambda x: -x**2 / 2
# num_samples = 3000
# iid_samples = np.random.normal(size=num_samples)

# kernel_samples = np.zeros(num_samples)
# for i in range(num_samples):
#     kernel_samples[i] = RWMH_exploration_kernel(log_gamma, iid_samples[i], 4000)[-1]

# ks_result = ks_2samp(iid_samples, kernel_samples)
# print("Kernel test p-value:", ks_result.pvalue)


## Toy example
def log_var_family(mu, sigma):
    return lambda x: -0.5 * ((x-mu) / sigma)**2

log_target = lambda x: -0.5 * ((x-10)/0.1)**2  ## N(10,0.01) target

num_chains = 5
initial_state = [0.1] * num_chains
num_tuning_rounds = 16
initial_phi = (0,0.5)


samples = variational_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target, log_var_family, initial_phi)
print("The mean is:", np.mean(samples))
print("The var is:", np.var(samples))
        
        
        
        
        
    