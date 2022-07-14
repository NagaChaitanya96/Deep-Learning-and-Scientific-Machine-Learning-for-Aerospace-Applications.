using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators, DomainSets
# Method of Manufactured Solutions: exact solution

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq  = Dt(u(t,x)) ~ -u(t,x)*Dx(u(t,x)) + (0.01/pi)*Dxx(u(t,x))

# Initial and boundary conditions
bcs = [u(0,x) ~ -sin(2*pi*x),
       u(t,0.) ~ 0.,
       u(t,1.) ~ 0.,]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]

# PDE system
@named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

# Method of lines discretization
dx = 0.05
# order = 2
discretization = MOLFiniteDifference([x=>dx],t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat = 0.05)

# Plot results and compare with exact solution
ts = [infimum(d.domain):dx:supremum(d.domain) for d in domains][2]
xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]

########################################################
T= sol.t[2:end-1]
U = sol.u
ts,xs = [0:0.05:1.0,0.0:0.05:1.0]

using Plots
plot(xs,U[end])
plt = plot(;title= "BurgersEquation for all time intervals",legend=nothing)
for i in 1:size(U,1)
    plot!(U[i])
end
display(plt)
savefig("BurgersEquation using DiffEqOperators.png")
