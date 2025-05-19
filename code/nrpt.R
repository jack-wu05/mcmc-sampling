set.seed(1234)

############# HYPERPARAMS & UTILITIES FOR HAMILTONIAN MONTE CARLO
epsilon = 0.05
L = 8

kick = function(s, gradient) {
  x = s[[1]]
  p = s[[2]]
  c(x, p + epsilon * gradient(x) / 2)
}

drift = function(s) {
  x = s[[1]]
  p = s[[2]]
  c(x + epsilon * p, p)
}

flip = function(s) {
  x = s[[1]]
  p = s[[2]]
  c(x, -p)
}

hmc_proposal = function(s, gradient) {
  for (i in 1:L) {
    s = kick(s, gradient)
    s = drift(s)
    s = kick(s, gradient)
  }
  flip(s)
}

hamiltonian = function(s, log_gamma) {
  x = s[[1]]
  p = s[[2]]
  
  -log_gamma(x) + 0.5*p^2
}

############# HAMILTONIAN MONTE CARLO KERNEL
hmc_exploration_kernel = function(initial_x, n_iteration, gradient, log_gamma) {
  samples = numeric(n_iteration)
  current_x = list(initial_x, rnorm(1))
  
  for (i in 1:n_iteration) {
    current_x[[2]] = rnorm(1)
    proposed_x = hmc_proposal(current_x, gradient)
    
    current_hamiltonian = hamiltonian(current_x, log_gamma)
    proposed_hamiltonian = hamiltonian(proposed_x, log_gamma)
    
    proportion = min(1, exp(current_hamiltonian - proposed_hamiltonian))
    
    if (runif(1) < proportion) {
      current_x = proposed_x
    }
    
    samples[i] = current_x[[1]]
    
  }
  return(samples)
}

############# CONSTRUCT ANNEALING PATH
path = function(betas, log_reference, log_target) {
  num_distributions = length(betas)
  temp = vector("list", num_distributions)
  
  for (i in 1:num_distributions) {
    beta = betas[i]
    temp[[i]] = local({
      b = beta
      function(x) {
        log_reference(x)*(1-b) + log_target(x)*b
      }
    })
  }
  return(temp)
}

############# COMPUTE ACCEPTANCE RATIO
alpha = function(log_dist1, log_dist2, state1, state2) {
  log_alpha = log_dist1(state2) + log_dist2(state1) -
    log_dist1(state1) - log_dist2(state2)
  return(min(1, exp(log_alpha)))
}

############# NRPT ALGORITHM
vanilla_NRPT = function(initial_state, betas, log_annealing_path, num_iterations, gradients) {
  num_distributions = length(betas)
  
  return_vector_of_vector_of_x = vector("list", num_iterations) 
  reject_rates = numeric(num_distributions)
  x_at_tminus1 = initial_state
  x_at_t = numeric(num_distributions)
  
  for (t in 1:num_iterations) {
    for (init in 1:num_distributions) {
      gradient = gradients[[init]]
      log_gamma = log_annealing_path[[init]]
      
      x_at_t[init] = hmc_exploration_kernel(x_at_tminus1[init], 1, gradient, log_gamma)
    }
    temp_alpha_vector = numeric(num_distributions)
    
    for (i in 1:(num_distributions-1)) {
      temp_alpha_vector[i] = alpha(log_annealing_path[[i]], log_annealing_path[[i+1]], x_at_t[i], x_at_t[i+1])
      reject_rates[i] = reject_rates[i] + (1 - temp_alpha_vector[i])/num_iterations
      U = runif(1)
      
      if (((i%%2==0 & t%%2==0) || (i%%2==1 & t%%2==1)) & (U <= temp_alpha_vector[i])) {
        x_at_t[c(i,i+1)] = x_at_t[c(i+1,i)]
      }
    }
    x_at_tminus1 = x_at_t
    return_vector_of_vector_of_x[[t]] = x_at_tminus1
  }
  
  return(
    list(reject_rates = reject_rates, samples = return_vector_of_vector_of_x)
  )
}



############# USAGE
log_reference = function(x) -2*(x^2)   ##un-normalized N(0,0.25)
log_target = function(x) -(x^2)/2     ##un-normalized N(0,1)
annealing_sched = c(0, 0.5, 1)
anneal_path = path(annealing_sched, log_reference, log_target)
initial_state = rep(0.1, length(annealing_sched))
num_iterations = 10
gradients = list(
  function(x) -4*x,
  function(x) -(5*x)/2,
  function(x) -x
)

result = vanilla_NRPT(initial_state, annealing_sched, anneal_path, num_iterations, gradients)

print(paste("The mean is:", mean(sapply(result$samples, function(chain) tail(chain, 1)))))
print(paste("GCB is:", sum(result$reject_rates)))
print(paste("Var is:", var(sapply(result$samples, function(chain) tail(chain, 1)))))


############# TESTING KERNEL
##Consider a Kolmogorov-Smirnov test for {iid N(0,1) samples} and 
##                                            {Kernel(iid N(0,1) samples)}

gradient = function(x) -x
log_gamma = function(x) -(x^2)/2
num_samples = 3000
iid_samples = numeric(num_samples)

for (i in 1:num_samples) {
  iid_samples[i] = rnorm(1)
}

kernel_samples = numeric(num_samples)
for (i in 1:num_samples) {
  result = hmc_exploration_kernel(iid_samples[i], 1000,
                                             gradient, log_gamma)
  kernel_samples[i] = result[length(result)]
}

ks.test(iid_samples,kernel_samples)

##Output: p-value = 0.692
##Accept null: The two sets are from the same distribution
##We're good!



############# TESTING DIFFERENT TEMPERATURES
log_reference = function(x) -2*(x^2)   ##un-normalized N(0,0.25)
log_target = function(x) -(x^2)/2     ##un-normalized N(0,1)
annealing_sched = c(0, 0.5, 1)
anneal_path = path(annealing_sched, log_reference, log_target)






#debug at each temperature
#use N(10,-0.1)
#change reference




