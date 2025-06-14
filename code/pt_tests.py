import numpy as np
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import nrpt
import tree_pt
import var_pt



def log_var_family(mu, cov):
    return lambda x: multivariate_normal.logpdf(np.array(x), mean=mu, cov=cov)

def plot(points1, points2, points3, points4, metric):
    x1,y1 = zip(*points1)  
    x2,y2 = zip(*points2)
    x3,y3 = zip(*points3)
    x4,y4 = zip(*points4)
      
    plt.figure()
    plt.plot(x1,y1, marker='o', color='b', label='Tree')
    plt.plot(x2,y2, marker ='o', color='r', label='Dense Gaussian')
    plt.plot(x3,y3, marker='o', color='g', label='Mean-Field Gaussian')
    plt.plot(x4,y4, marker='o', color='c', label='Fixed N(0,I)')
    plt.xlabel('Tuning Round (r)')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()
    

### Toy example
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
num_tuning_rounds = 12
initial_phi = (np.zeros(d), np.eye(d))



### True value
samples = []
for i in range(100000):
    sample = np.random.multivariate_normal(mean=mean, cov=cov)
    samples.append(sample)
print("True mean:", np.mean(samples, axis=0))
print("True variance:", np.var(samples, axis=0))
print()

### Dense
samples, rates, Lambda_vs_r_dense, NumRestarts_vs_r_dense, Tau_vs_r_dense= var_pt.variational_PT_with_RWMH(initial_state, num_chains, 
                                                                                                           num_tuning_rounds, log_target,
                                                                                                           log_var_family, initial_phi,
                                                                                                           diagonal=False, variation=True)
print("----------------------")
print()
print("Final reject rates:",rates)
print("(Dense Gaussian) Estimated mean:", np.mean(samples, axis=0))
print("(Dense Gaussian) Estimated variance:", np.var(samples, axis=0))
print()


### Mean-field
samples, rates, Lambda_vs_r_diag, NumRestarts_vs_r_diag, Tau_vs_r_diag = var_pt.variational_PT_with_RWMH(initial_state, num_chains, 
                                                                                                         num_tuning_rounds, log_target,
                                                                                                         log_var_family, initial_phi, 
                                                                                                         diagonal=True, variation=True)
print("----------------------")
print()
print("Final reject rates:",rates)
print("(Mean-Field Gaussian) Estimated mean:", np.mean(samples, axis=0))
print("(Mean-Field Gaussian) Estimated variance:", np.var(samples, axis=0))
print()


### Fixed N(0,I)
samples, rates, Lambda_vs_r_fixed, NumRestarts_vs_r_fixed, Tau_vs_r_fixed = var_pt.variational_PT_with_RWMH(initial_state, num_chains, 
                                                                                                            num_tuning_rounds, log_target,
                                                                                                            log_var_family, initial_phi, 
                                                                                                            diagonal=False, variation=False)
print("----------------------")
print()
print("Final reject rates:",rates)
print("(Fixed Standard Gaussian) Estimated mean:", np.mean(samples, axis=0))
print("(Fixed Standard Gaussian) Estimated variance:", np.var(samples, axis=0))
print()

### Tree
samples, rates, Lambda_vs_r_TREE, NumRestarts_vs_r_TREE, Tau_vs_r_TREE = tree_pt.tree_PT_with_RWMH(initial_state, num_chains, 
                                                                                                   num_tuning_rounds, log_target)
print("----------------------")
print()
samples = np.array(samples)
print("(Tree) Estimated mean:", np.mean(samples, axis=0))
print("(Tree) Estimated variance:", np.var(samples, axis=0))


### Figure 1
plot(Lambda_vs_r_TREE, Lambda_vs_r_dense, Lambda_vs_r_diag, Lambda_vs_r_fixed, metric='GCB (\u03BB)')

### Figure 2
plot(Tau_vs_r_TREE, Tau_vs_r_dense, Tau_vs_r_diag, Tau_vs_r_fixed, metric='Restart Rate (\u03C4)')

### Figure 3
plot(NumRestarts_vs_r_TREE, NumRestarts_vs_r_dense, NumRestarts_vs_r_diag, NumRestarts_vs_r_fixed, metric='Num Restarts')
