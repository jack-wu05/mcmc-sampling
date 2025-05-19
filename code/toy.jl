using Pkg
Pkg.add(["Pigeons", "Turing"])
Pkg.add(["MCMCChains", "StatsPlots", "PlotlyJS"])
Pkg.add(["PairPlots", "CairoMakie"])
Pkg.add("DynamicPPL")
Pkg.add("MPI")
using DynamicPPL

using Turing
@model function coinflip(n, y)
    p1 ~ Uniform(0.0, 1.0)
    p2 ~ Uniform(0.0, 1.0)
    y ~ Binomial(n, p1 * p2)
    return y
end
model = coinflip(100000, 50000)

using Pigeons
pt = pigeons(
    target = TuringLogPotential(model),
    record = [
        traces, online, round_trip,
        Pigeons.timing_extrema,
        Pigeons.allocation_extrema],
    variational = GaussianReference(),  # variational reference
    n_chains = 5,
    seed = 2)

# using MPI
### DISTRIBUTED IMPLEMENTATION
# result = pigeons(
#     target = TuringLogPotential(model),
#     checkpoint = true,
#     on = ChildProcess(
#         n_local_mpi_processes = 4))
# pt = Pigeons.load(result)

using MCMCChains, StatsPlots, PlotlyJS
plotlyjs()
samples = Chains(pt)

my_plot = StatsPlots.plot(samples)
display(my_plot)

using PairPlots, CairoMakie
my_plot = PairPlots.pairplot(samples)
display(my_plot)


println("Z: $(stepping_stone(pt))")
using Statistics
println("Mean: $(mean(pt))")
println("Var: $(var(pt))")
println("GCB: $(Pigeons.global_barrier(pt))")