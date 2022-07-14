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
bcs = [u(0,x) ~ -sin(2pi*x),
       u(t,0.) ~ 0.,
       u(t,1.) ~ 0.,]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]

# PDE system
@named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

# Method of lines discretization
dx = 0.1
# order = 2
discretization = MOLFiniteDifference([x=>dx],t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat = 0.1)

# Plot results and compare with exact solution
ts = [infimum(d.domain):dx:supremum(d.domain) for d in domains][2]
xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]

########################################################
t= sol.t
using Plots
plt = plot()
U = sol.u
plot(t,U)
for i in 1:length(t)
    plot!(sol.t,sol.u[i],label="Numerical, t=$(t[i])")
    #scatter!(x, u_exact(x, t[i]),label="Exact, t=$(t[i])")
end


display(plt)



savefig("plot.png")



ts,xs = [0:0.05:1.0,-1:0.05:1.0]
u_predict_contourf = reshape([first(phi([t,x],res.minimizer)) for t in ts for x in xs] ,length(xs),length(ts))
plot(ts, xs, u_predict_contourf, linetype=:contourf,title = "predict")

u_predict = [[first(phi([t,x],res.minimizer)) for x in xs] for t in ts ]
p1= plot(xs, u_predict[3],title = "t = 0.1");
p2= plot(xs, u_predict[11],title = "t = 0.5");
p3= plot(xs, u_predict[end],title = "t = 1");
plot(p1,p2,p3)
