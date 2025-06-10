import numpy as np
import warnings
from scipy.stats import ks_2samp
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator
import pints
import pints.toy
from scipy.stats import multivariate_normal

import nrpt

import matplotlib.pyplot as plt

s = 0.7

def RWMH_exploration_kernel(log_gamma, initial_x, num_iters):
    curr_point = initial_x
    d = 4
    samples = np.zeros((num_iters, d))
    
    for i in range(num_iters):
        new_point = np.zeros(d)
        
        for j in range(d):
            new_point[j] = np.random.normal(curr_point[j], s, 1)
            
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
        lambda_hat[i] = sum(reject_rates[j] for j in range(i))
    
    # ## PLOT FOR DEBUGGING
    # plt.figure()
    # plt.plot(schedule, lambda_hat, marker='o')
    # plt.grid(True)
    # plt.show()
    
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
        new_schedule[p] = bisect(0,1,func,epsilon=0.0001)
    
    return new_schedule

    
def update_dense_reference(samples):
    samples = np.asarray(samples)
    return (np.mean(samples, axis=0), np.cov(samples.T))

def update_diagonal_reference(samples):
    samples = np.asarray(samples)
    
    variances = np.zeros(samples.shape[1])
    for i in range(samples.shape[1]):
        variances[i] = np.var(samples[:,i])
        
    return (np.mean(samples, axis=0), np.diag(variances))


def vanilla_NRPT_with_RWMH(initial_state, betas, log_annealing_path, num_iterations, variation):
    num_distributions = len(betas)
    d = 4

    samples = []
    reject_rates = np.zeros(num_distributions-1)
    
    x_at_tminus1 = initial_state.copy()
    x_at_t = np.zeros((num_distributions, d))

    for t in range(num_iterations):
        for init in range(num_distributions):
            log_gamma = log_annealing_path[init]

            x_at_t[init] = RWMH_exploration_kernel(log_gamma, x_at_tminus1[init], 15)[-1]
        
        temp_alpha_vector = np.zeros(num_distributions)

        for i in range(num_distributions-1):
            temp_alpha_vector[i] = nrpt.alpha(log_annealing_path[i], log_annealing_path[i+1], x_at_t[i], x_at_t[i+1])
            
            if variation == False:
                reject_rates[i] = reject_rates[i] + (1 - temp_alpha_vector[i]) / 1000
            else:
                reject_rates[i] = reject_rates[i] + (1 - temp_alpha_vector[i]) / 100
            
            if ((i%2==0 and t%2==0) or (i%2==1 and t%2==1)) and (np.random.rand() <= temp_alpha_vector[i]):
                x_at_t[i], x_at_t[i+1] = x_at_t[i+1], x_at_t[i]

        x_at_tminus1 = x_at_t.copy()
        samples.append(x_at_tminus1.copy())

    return {
        "reject_rates": reject_rates,
        "samples": samples
    }
    
    
## Variational PT implementation!
def variational_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target, log_var_family, initial_phi, diagonal, variation):
    schedule = np.linspace(0, 1, num_chains)
    curr_state = initial_state
    curr_phi = initial_phi
    
    Lambda_vs_r_points = []
    
    toReturn = []
    
    for r in range(4, num_tuning_rounds):
        print("-----","TUNING ROUND", r,"-----")
        
        T = 2**r
        
        curr_var = log_var_family(curr_phi[0], curr_phi[1])
        path = nrpt.path(schedule, curr_var, log_target)
        
        result = vanilla_NRPT_with_RWMH(curr_state, schedule, path, T, variation)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        if r >= 5: 
            for chain in samples: toReturn.append(chain[-1])
        
        schedule = update_schedule(reject_rates, schedule)
        
        if variation == True:
            if diagonal == True:
                curr_phi = update_diagonal_reference([chain[-1] for chain in samples])
            else:
                curr_phi = update_dense_reference([chain[-1] for chain in samples])
            
        curr_state = samples[-1]
        
        print("Reject rates:",reject_rates)
        print("Schedule:",schedule)
        
        ## Collect points for Lambda vs. r plot
        Lambda_vs_r_points.append([r, np.sum(reject_rates / (1 - reject_rates))])
    
    return toReturn, reject_rates, Lambda_vs_r_points


### Kolmogorov-Smirnov test for kernel pi-invariance using N(0,1)
# log_gamma = lambda x: -x**2 / 2
# num_samples = 3000
# iid_samples = np.random.normal(size=num_samples)

# kernel_samples = np.zeros(num_samples)
# for i in range(num_samples):
#     kernel_samples[i] = RWMH_exploration_kernel(log_gamma, iid_samples[i], 4000)[-1]

# ks_result = ks_2samp(iid_samples, kernel_samples)
# print("Kernel test p-value:", ks_result.pvalue)


### Toy examples
def log_var_family(mu, cov):
    return lambda x: multivariate_normal.logpdf(np.array(x), mean=mu, cov=cov)

d = 4
mean = np.array([1,2,3,4])
cov = np.array([
    [1.0,  0.8,  0.5,  0.3],
    [0.8,  1.5,  0.6,  0.4],
    [0.5,  0.6,  2.0,  0.7],
    [0.3,  0.4,  0.7,  1.8]
])
log_target = lambda x: multivariate_normal.logpdf(np.array(x), mean=mean, cov=cov)

num_chains = 15
initial_state = [[0.25, 0.25, 0.25, 0.25]] * num_chains
num_tuning_rounds = 10
initial_phi = (np.zeros(d), np.eye(d))

## Dense
samples, rates, Lambda_vs_r_points_VAR_dense = variational_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target, 
                                                                        log_var_family, initial_phi, diagonal=False, variation=True)
print("Final reject rates:",rates)
print("(Dense Gaussian) Estimated mean:", np.mean(samples, axis=0))
print("(Dense Gaussian) Estimated variance:", np.var(samples, axis=0))


## Mean-field
samples, rates, Lambda_vs_r_points_VAR_diagonal = variational_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target, 
                                                                           log_var_family, initial_phi, diagonal=True, variation=True)
print("Final reject rates:",rates)
print("(Mean-Field Gaussian) Estimated mean:", np.mean(samples, axis=0))
print("(Mean-Field Gaussian) Estimated variance:", np.var(samples, axis=0))
        

## Fixed N(0,I)
samples, rates, Lambda_vs_r_points_fixed = variational_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target,
                                                                    log_var_family, initial_phi, diagonal=False, variation=False)
print("Final reject rates:",rates)
print("(Fixed Standard Gaussian) Estimated mean:", np.mean(samples, axis=0))
print("(Fixed Standard Gaussian) Estimated variance:", np.var(samples, axis=0))
        