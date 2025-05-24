import numpy as np
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings("ignore")

##### To come: autodiff, Z estimate, automatic schedule tuning


## Hyperparameters for HMC kernel
np.random.seed(1234)
epsilon = 0.05
L = 10

## Utility for HMC kernel
def kick(s, gradient):
    x,p = s
    return (x, p + epsilon * gradient(x)/2)

## Utility for HMC kernel
def drift(s):
    x,p = s
    return (x + epsilon * p, p)

## Utility for HMC kernel
def flip(s):
    x,p = s
    return (x, -p)

## Utility for HMC kernel
def hmc_proposal(s, gradient):
    for i in range(L):
        s = kick(s, gradient)
        s = drift(s)
        s = kick(s, gradient)
    return flip(s)

## Utility for HMC kernel
def hamiltonian(s, log_gamma):
    x,p = s
    return -log_gamma(x) + 0.5 * p**2


## HMC kernel
def hmc_exploration_kernel(initial_x, n_iteration, gradient, log_gamma):
    samples = np.zeros(n_iteration)
    current_x = (initial_x, np.random.normal())

    for i in range(n_iteration):
        current_x = (current_x[0], np.random.normal())
        proposed_x = hmc_proposal(current_x, gradient)

        current_hamiltonian = hamiltonian(current_x, log_gamma)
        proposed_hamiltonian = hamiltonian(proposed_x, log_gamma)

        proportion = min(1, np.exp(current_hamiltonian - proposed_hamiltonian))

        if np.random.rand() < proportion:
            current_x = proposed_x
        
        samples[i] = current_x[0]
    
    return samples

## Helper for path
def make_density(beta, log_reference, log_target):
    return lambda x: log_reference(x)*(1-beta) + log_target(x)*beta

## Construct annealing path
def path(betas, log_reference, log_target):
    return [make_density(beta, log_reference, log_target) for beta in betas]

## Compute acceptance probability
def alpha(log_dist1, log_dist2, state1, state2):
    log_alpha = log_dist1(state2) + log_dist2(state1) - log_dist1(state1) - log_dist2(state2)
    return min(1, np.exp(log_alpha))

## NRPT implementation!
def vanilla_NRPT_with_HMC(initial_state, betas, log_annealing_path, num_iterations, gradients):
    num_distributions = len(betas)

    samples = []
    reject_rates = np.zeros(num_distributions)
    x_at_tminus1 = initial_state.copy()
    x_at_t = np.zeros(num_distributions)

    for t in range(num_iterations):
        for init in range(num_distributions):
            gradient = gradients[init]
            log_gamma = log_annealing_path[init]

            x_at_t[init] = hmc_exploration_kernel(x_at_tminus1[init], 1, gradient, log_gamma)
        
        temp_alpha_vector = np.zeros(num_distributions)

        for i in range(num_distributions-1):
            temp_alpha_vector[i] = alpha(log_annealing_path[i], log_annealing_path[i+1], x_at_t[i], x_at_t[i+1])
            reject_rates[i] = reject_rates[i] + (1 - temp_alpha_vector[i])/num_iterations
            
            if ((i%2==0 and t%2==0) or (i%2==1 and t%2==1)) and (np.random.rand() <= temp_alpha_vector[i]):
                x_at_t[i], x_at_t[i+1] = x_at_t[i+1], x_at_t[i]

        x_at_tminus1 = x_at_t.copy()
        samples.append(x_at_tminus1.copy())

    return {
        "reject_rates": reject_rates,
        "samples": samples
    }




# ## Toy example
# log_reference = lambda x: -2 * x**2     ## N(0,0.25) reference
# log_target = lambda x: -x**2 / 2    ## N(0,1) target
# annealing_sched = [0, 0.5, 1]
# anneal_path = path(annealing_sched, log_reference, log_target)
# initial_state = [0.1] * len(annealing_sched)
# num_iterations = 10000
# gradients = [
#     lambda x: -4 * x,
#     lambda x: -2.5 * x,
#     lambda x: -x
# ]    

# result = vanilla_NRPT_with_RWMH(initial_state, annealing_sched, anneal_path, num_iterations, gradients)
# last_samples = [chain[-1] for chain in result["samples"]]
# print("The mean is:", np.mean(last_samples))
# print("GCB estimate is:", np.sum(result["reject_rates"]))
# print("Var is:", np.var(last_samples))



# ## Kolmogorov-Smirnov test for kernel pi-invariance using N(0,1)
# gradient = lambda x: -x
# log_gamma = lambda x: -x**2 / 2
# num_samples = 200
# iid_samples = np.random.normal(size=num_samples)

# kernel_samples = np.zeros(num_samples)
# for i in range(num_samples):
#     kernel_samples[i] = hmc_exploration_kernel(iid_samples[i], 400, gradient, log_gamma)[-1]

# ks_result = ks_2samp(iid_samples, kernel_samples)
# print("Kernel test p-value:", ks_result.pvalue)



# ## Run Kolmogorov-Smirnov for each temperature for a toy example
# def debug_PT(gradients, log_gammas, num_samples, num_iterations_per_sample, sigmas):
#     num = len(log_gammas)
#     for i in range(num):
#         gradient = gradients[i]
#         log_gamma = log_gammas[i]
#         sigma = sigmas[i]
        
#         iid_samples = np.random.normal(loc=0, scale=sigma, size=num_samples)

#         kernel_samples = np.zeros(num_samples)
#         for j in range(num_samples):
#             kernel_samples[j] = hmc_exploration_kernel(iid_samples[j], num_iterations_per_sample, gradient, log_gamma)[-1]
            
#         ks_result = ks_2samp(iid_samples, kernel_samples)
#         print("Path", i, "test p-value:", ks_result.pvalue)

# sigmas = [0.5, np.sqrt(10)/5 ,1]
# debug_PT(gradients, anneal_path, 2000, 4000, sigmas)



# ## Visualize a non-linear geometric sequence in [0,1] for better betas
# space = np.linspace(0, 1, num=50)
# for i in range(len(space)):
#     print(1 - np.exp(-space[i]))
# print("=================")
# print(space)


## Non-trivial use case for NRPT
# log_reference = lambda x: -x**2 / 2     ## N(0,1) reference
# log_target = lambda x: -500 * (x-10)**2     ## N(10,0.001) target
# annealing_sched = [0, 0.01, 0.05, 0.15, 0.25, 0.40, 0.60, 0.80, 1]
# anneal_path = path(annealing_sched, log_reference, log_target)
# initial_state = [0.1] * len(annealing_sched)
# num_iterations = 10000
# gradients = [
#     lambda x: -x,
#     lambda x: -10.99 * x + 100,
#     lambda x: -50.95 * x + 500,
#     lambda x: -150.85 * x + 1500,
#     lambda x: -250.75 * x + 2500,
#     lambda x: -400.6 * x + 4000,
#     lambda x: -600.4 * x + 6000,
#     lambda x: -800.2 * x + 8000,
#     lambda x: -1000 * x + 10000
# ]    

# result = vanilla_NRPT_with_HMC(initial_state, annealing_sched, anneal_path, num_iterations, gradients)
# last_samples = [chain[-1] for chain in result["samples"]]
# print("The mean is:", np.mean(last_samples))
# print("GCB estimate is:", np.sum(result["reject_rates"]))
# print("Var is:", np.var(last_samples))





