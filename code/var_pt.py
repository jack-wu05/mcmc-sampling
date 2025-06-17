import numpy as np
import numpy.linalg as linalg
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

import nrpt


## Metropolis-Hastings random walk exploration kernel
def RWMH_exploration_kernel(log_gamma, initial_x, num_iters):
    curr_point = initial_x
    d = 4
    samples = np.zeros((num_iters, d))
    
    for i in range(num_iters):
        new_point = np.zeros(d)
        
        for j in range(d):
            new_point[j] = np.random.normal(curr_point[j], 0.6, 1)
            
        p = min(1, np.exp(log_gamma(new_point)-log_gamma(curr_point)))
    
        if np.random.rand() < p:
            curr_point = new_point
    
        samples[i] = curr_point
        
    return samples


## Utility for update_schedule
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


## Utility for update_schedule
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
    

## Automatic schedule tuning
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


## Update rule for dense reference
def update_dense_reference(samples):
    samples = np.asarray(samples)
    return (np.mean(samples, axis=0), np.cov(samples.T))


## Update rule for mean-field reference
def update_diagonal_reference(samples):
    samples = np.asarray(samples)
    
    variances = np.zeros(samples.shape[1])
    for i in range(samples.shape[1]):
        variances[i] = np.var(samples[:,i])
        
    return (np.mean(samples, axis=0), np.diag(variances))


## Compute forward KL(G0||G1), where Gaussian1 is reference, Gaussian0 is target
def kl_div(mu0, Sigma0, mu1, Sigma1):
    term1 = np.log(linalg.det(Sigma1) / linalg.det(Sigma0))
    term2 = -1 * Sigma0.shape[0]
    term3 = np.trace(linalg.inv(Sigma1) @ Sigma0)
    term4 = (mu1 - mu0).T @ linalg.inv(Sigma1) @ (mu1 - mu0)
    
    return 0.5 * (term1 + term2 + term3 + term4)
    

## Utility for variational_PT_with_RWMH
def vanilla_NRPT_with_RWMH(initial_state, betas, log_annealing_path, num_iterations):
    num_distributions = len(betas)
    d = 4
    restarts = 0

    samples = []
    reject_rates = np.zeros(num_distributions-1)
    
    x_at_tminus1 = initial_state.copy()
    x_at_t = np.empty(num_distributions, dtype=object)

    for t in range(num_iterations):
        for init in range(num_distributions):
            log_gamma = log_annealing_path[init]

            curr_colour = x_at_tminus1[init][1]
            x_at_t[init] = (RWMH_exploration_kernel(log_gamma, x_at_tminus1[init][0], 20)[-1], curr_colour)
        
        temp_alpha_vector = np.zeros(num_distributions)

        for i in range(num_distributions-1):
            temp_alpha_vector[i] = nrpt.alpha(log_annealing_path[i], log_annealing_path[i+1], x_at_t[i][0], x_at_t[i+1][0])
            reject_rates[i] = reject_rates[i] + (1 - temp_alpha_vector[i]) / num_iterations
            
            if ((i%2==0 and t%2==0) or (i%2==1 and t%2==1)) and (np.random.rand() <= temp_alpha_vector[i]):
                x_at_t[i], x_at_t[i+1] = x_at_t[i+1], x_at_t[i]
                
                if i == num_distributions-2:
                    if (x_at_t[i+1][1] == "b"):
                        x_at_t[i+1] = (x_at_t[i+1][0], "w")
                        restarts += 1
                        
                if i == 0:
                    x_at_t[0] = (x_at_t[0][0], "b")
                    x_at_t[1] = (x_at_t[1][0], "b")
                

        x_at_tminus1 = x_at_t.copy()
        samples.append(x_at_tminus1.copy())

    return restarts, {
        "reject_rates": reject_rates,
        "samples": samples
        }
    
    
##### Variational PT implementation!
# colour "b" (black): has hit the reference
# colour "w" (white): was black, and has hit the target
# colour "g" (grey): neither
def variational_PT_with_RWMH(initial_state, num_chains, num_tuning_rounds, log_target, log_var_family, 
                             initial_phi, diagonal, variation, target_mu, target_Sigma):
    schedule = np.linspace(0, 1, num_chains)
    curr_state = [(point, "g") for point in initial_state]
    curr_phi = initial_phi
    d = 4
    
    Lambda_vs_r = []
    RestartRate_vs_r = []
    NumRestarts_vs_r = []
    kl_vs_r = []
    restarts_vs_cost = []
    
    toReturn = []
    
    for r in range(6, num_tuning_rounds):
        print("-----","TUNING ROUND", r,"-----")
        
        curr_state = [(tup[0], "g") for tup in curr_state]
        T = 2**r
        cost = 0
        
        curr_var = log_var_family(curr_phi[0], curr_phi[1])
        path = nrpt.path(schedule, curr_var, log_target)
        

        restarts, result = vanilla_NRPT_with_RWMH(curr_state, schedule, path, T)
        reject_rates = result["reject_rates"]
        samples = result["samples"]
        
        if r >= 5: 
            for chain in samples: toReturn.append(chain[-1][0])
        
        schedule = update_schedule(reject_rates, schedule)
        
        
        ## Collect points for KL vs. r plot
        kl_vs_r.append([r, kl_div(target_mu, target_Sigma, curr_phi[0], curr_phi[1])])
        
        if variation == True:
            if diagonal == True:
                curr_phi = update_diagonal_reference([chain[-1][0] for chain in samples])
            else:
                curr_phi = update_dense_reference([chain[-1][0] for chain in samples])
        
        curr_state = samples[-1]
        
        if variation == False:
            cost += T*num_chains*d
        else:
            if diagonal == True:
                cost += d + T*num_chains*d
            else:
                cost += d**3 + T*num_chains * d**2
        
        print("Reject rates:",reject_rates)
        print("Schedule:",schedule)
        print()
        
        ## Collect points for Lambda vs. r plot
        Lambda_vs_r.append([r, np.sum(reject_rates / (1 - reject_rates))])
        
        ## Collect points for Num Restarts vs. r plot
        NumRestarts_vs_r.append([r, restarts])
        
        ## Collect points for Tau vs. r plot
        RestartRate_vs_r.append([r, restarts/T])
        
        ## Collect points for num restarts vs. O(cost) plot
        restarts_vs_cost.append([cost, restarts])
    
    
    return toReturn, reject_rates, Lambda_vs_r, NumRestarts_vs_r, RestartRate_vs_r, kl_vs_r, restarts_vs_cost
