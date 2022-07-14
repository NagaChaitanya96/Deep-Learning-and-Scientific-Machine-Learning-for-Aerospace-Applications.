""" Using Data to estimate parameters for Mechanistic models with sigmoid approximate function
            Data is send into additional loss function to train our model""";

##
using ModelingToolkit, NeuralPDE, DiffEqFlux, DomainSets
import ModelingToolkit: infimum, supremum, Interval
using GalacticOptim
@parameters t,a,b,c,d,e
@variables h(..), v(..),u(..)

Dt = Differential(t)

eqn = [Dt(h(t)) ~ v(t),
      Dt(v(t)) ~ -1.5 + (a+b*t+c*t^2+d*t^4+e*t^7),
      Dt(u(t)) ~ b + 2*c*t + 4*d*t^3 + 7*e*t^6]
      tf = 4.16

      bcs = [h(0.) ~ 10.,
              v(0.) ~ -2., # its decelerating
              u(0.) ~ 0.,
              u(tf) ~ 3.,
              h(tf) ~ 0.,
              v(tf) ~ 0.,
              ]

domains = [t in Interval(0.,tf)]
input_ =1
chain = [FastChain(FastDense(input_,10,σ),FastDense(10,10,σ),FastDense(10,1)) for _ in 1:3]

initθ = map(c -> Float64.(DiffEqFlux.initial_params(c)),chain)

flat_initθ = reduce(vcat,initθ)
dt = 0.05
strategy = NeuralPDE.GridTraining(dt)
indvars = [t]
depvars =[h,v,u]
ps = [a,b,c,d,e]
defaults = Dict([p => 05 for p in ps])


function AnalyticalSol(t,v0,h0)

  tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
  ŝ = tf̂/2 + v0/3
  v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
  ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
  û(t) = (t<=ŝ) ? 0 : 3
  #println("The values are : $([tf̂, ŝ ,v̂(t) , ĥ(t), û(t)])")
  return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
end
ts = 0.0 : dt : tf
hdata = [AnalyticalSol(a,-2.,10.)[4] for a in ts]
udata =  [AnalyticalSol(a,-2.,10.)[5] for a in ts]
vdata = [AnalyticalSol(a,-2.,10.)[3] for a in ts]
u_ = [hdata'; vdata'; udata']
t_ = hcat(collect(ts)...)
len = length(t_)

function additional_loss(phi, θ , p)
  uₚ(i) = ([first(phi[i](t,θ[sep[i]])) for t in t_]) # θ = res.minimizer
  """ This is the l2 norm for all the constraints""";
  add_loss1 = sum(sum(abs2.(uₚ(i) .- u_[[i], :]))/len for i in 1:3)
  return add_loss1
end

discretization = PhysicsInformedNN(chain,strategy,param_estim = true ,additional_loss =additional_loss)
@named pde_system = PDESystem(eqn,bcs,domains,indvars,[h(t),v(t),u(t)],ps,defaults =defaults)
prob = discretize(pde_system,discretization)
cb = function(p,l)
  println("The loss is :",l)
  return false
end
initθ = discretization.init_params
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
res = GalacticOptim.solve(prob, BFGS();cb =cb,maxiters = 500)
#prob = remake(prob,u0 = res.minimizer)
#res = GalacticOptim.solve(prob, ADAM(0.01);cb =cb,maxiters = 1500)
minimizers = [res.minimizer[s] for s in sep]
phi = discretization.phi
u_predict = [[first(phi[i](t,minimizers[i])) for t in ts] for i in 1:3]
using Plots
p1 = plot(ts,[u_[i,:] for i in 1:3],yticks = -5:1:10.5,xticks = 0.0:0.5:4.2,title = "Analytical Result",xlims =(0.0,4.2))
p2 = plot(ts, u_predict, label = ["h(t)" "v(t)" "u(t)"],yticks = -5:1:10.5,xticks = 0.0:0.5:4.2, title = "NeuralPDE result",xlims = (0.0,4.2))
plot(p1,p2,plot_title = "Polynomial fit with Data")
p = res.minimizer[end-4:end]

#png("D:\\Chaitanya\\sciml\\Analysis so far\\ Using Data to estimate parameters for polynomial")
