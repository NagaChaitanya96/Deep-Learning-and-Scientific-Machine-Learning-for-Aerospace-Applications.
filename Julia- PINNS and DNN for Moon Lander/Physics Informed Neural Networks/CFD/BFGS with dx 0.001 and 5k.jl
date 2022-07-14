using NeuralPDE, ModelingToolkit, DiffEqFlux, GalacticOptim, Plots
using DifferentialEquations
using DiffEqOperators
using DomainSets
using OrdinaryDiffEq
using CUDA
##
""" Symbolic representation""";
@parameters t,x;
@variables u(..);

Dx = Differential(x);
Dt = Differential(t);
Dxx = Differential(x).^2;

eqn = Dt(u(t,x)) ~ Dxx(u(t,x));
bcs = [u(0,x) ~ cos(x),
        u(t,0) ~ exp(-t),
        u(t,1) ~ exp(-t)* cos(1)];

domains = [x ∈ Interval(0.0,1.0),
           t ∈ Interval(0.0,1.0)];
dx = 0.001       # grid training;

##
""" Defining NeuralNetwork """;;
N1 =16;
chain = FastChain(FastDense(2,N1,σ),FastDense(N1,N1,σ),FastDense(N1,N1,σ),FastDense(N1,1,σ));
initθ = Float64.(DiffEqFlux.initial_params(chain)) |>gpu;
discretization = PhysicsInformedNN(chain,GridTraining(dx);inital_params = initθ);
indvars = [t,x];
depvars = [u(t,x)];

@named pdesys = PDESystem(eqn,bcs,domains,indvars,depvars);
prob = discretize(pdesys,discretization);
cb_ = function (p,l)
    #println("loss: ", l , "losses: ", map(l -> l(p), lossfunctions))
    println("the current loss is : $l")
    return false
end;
opt = BFGS();
maxiters=5000;
res = GalacticOptim.solve(prob, opt; cb = cb_, maxiters=maxiters);
phi = discretization.phi

##
""" COMPARISION WITH EXACT SOLUTION """;
function L2(f::Vector{Vector{Float64}},g::Matrix{Float64},i)
        return abs(sum(g[:,i].- f[i]).^2)
end;

function L∞(f::Vector{Vector{Float64}},g::Matrix{Float64},i)
        return maximum(abs.(f[i].-g[:,i]))
end;

""" PLOTS""";
ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains];
u_predict = [[first(phi([t,x],res.minimizer)) for x in xs] for t in ts ];
""" EXACT SOLUTION """;
u_ex = (t,x) -> exp.(-t) * cos.(x);
u_exact = [u_ex(t,x) for x in xs, t in ts];

l∞ = [L∞(u_predict,u_exact,i) for i in 1 : length(u_predict)];
l2 = [L2(u_predict,u_exact,i) for i in 1 : length(u_predict)];
lbl="BFGS() with Iter =$(maxiters)"
plt1 = plot(ts,l∞,title =" L∞ norm for stepsize =$(dx)", label = lbl)
xlabel!("x")
ylabel!("Error")
png("D:\\l∞ norm at dx = $(dx),$(lbl)")

plt2 = plot(ts,l2,title =" L2 norm for stepsize =$(dx)", label = lbl);
xlabel!("x")
ylabel!("Error")

png("D:\\L2norm at dx = $(dx),$(lbl)")
