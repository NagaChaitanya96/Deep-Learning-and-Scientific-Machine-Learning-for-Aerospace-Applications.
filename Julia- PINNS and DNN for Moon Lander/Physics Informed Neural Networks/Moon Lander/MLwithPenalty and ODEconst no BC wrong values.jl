begin
      using ModelingToolkit
      using DiffEqFlux
      using NeuralPDE
      import ModelingToolkit: Interval
      using GalacticOptim
end

# ODE definition {{{

@parameters t,a,b
@variables h(..),v(..), u(..)

begin
      @parameters t,a,b
      @variables h(..),v(..), u(..)

      Dt = Differential(t)
      #  H, V are +ve upward direction , since flight is descending we have v0 = -2 as initial condition, and h0 =10.0 initially. As the aircraft reaches
      # the ground the Hfinal should be 0 and v final should reach 0. For this we need to minimize the energy consumption ∫Udt.

      # to solve this we are solving the system of ODE's and given the initial conditions. As a part of ModelingToolkit interface, we give the indvars and
      # depvars.
      eqn = [Dt(h(t)) ~ v(t),
                Dt(v(t)) ~ -1.5 + (a+b*t)]
               # Dt(u(t)) ~ b]

      v0= -2.; h0 = 10.
      tf = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))

      bcs = [h(0.) ~ h0,
             v(0.) ~ v0] # its deceleratin
             #u(0.) ~ 0.,
             #u(tf) ~ 3.,
             #h(tf) ~ 0.,
             #v(tf) ~ 0.]

      domains = [t in Interval(0.,tf)]
      indvars = [t]
      #depvars = [h(t), v(t), u(t)]
      depvars = [h(t), v(t)]
end

# Function for Analytical Solution {{{
function AnalyticalSol(t,v0,h0)
      tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
      ŝ = tf̂/2 + v0/3
      v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
      ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
      û(t) = (t<=ŝ) ? 0 : 3
      #println("The values are : $([tf̂, ŝ ,v̂(t) , ĥ(t), û(t)])")
      return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
end



begin
      cb = function(p,l)
              println("The loss is :",l)
              return false
      end
      # How do this penalty assures that H is always positive?
      #= Here we are minimizing the u21 value to go to zero. and for which we use up1 as the input, here its actually minimizing u21, for the
      corresponding up1 values. How does this effect H to become +ve? It can be anything and since we are taking absolute it will not make a
      difference.

      IF we need to have have H > 0; we need to make sure that trail solution after paramaterzing the NN will give us positive for all times.
      i.e we need to modify uₚ(i), and which includes modification of params, which is not in our control.

      We have different activation functions to control for parms , tanh,σ, leakyRelu,etc.
      ELse
      we need to set a constraint such that;
            if minimum uₚ(1) < 0
                  our params should be modified in such a way that minimum uₚ(1) always ≈ 0
      =#
      function Hpen(phi,θ,p)
                  uₚ(i) = ([first(phi[i](t,θ[sep[i]])) for t in t_]) # θ
                  up1 =  uₚ(1)
                  #u21 =  [(up1[i] >=0.0) ? up1[i] : up1[i] .+ exp(-2*up1[i].*t2[i]) for i in 1:length(t_)]
                  u21 =  [(up1[i] >=0.0) ? up1[i] : up1[i].^2 for i in 1:length(t_)]
                  J2 = 5e-3*sum(u21)
                  return J2
            end
end

begin
      parameters = [a,b]
      initGuess = Dict([p => 0.5 for p in parameters])
      # Check to see if there is an ODE System function
      @named pde_system = PDESystem(eqn,bcs,domains,indvars,depvars,parameters,defaults = initGuess)
      n = 10
      chain = [FastChain(FastDense(1 ,n,σ),FastDense(n,n,σ),FastDense(n,1)) for _ in 1:2]
      dt = 0.01;
      ts = 0.:dt:tf
      t_ = collect(ts)
      u21 = zeros(length(t_))
      t2 = t_
      grid_strategy = NeuralPDE.GridTraining(dt);
      #discretization = PhysicsInformedNN(chain,grid_strategy,param_estim = true, additional_loss = nothing)
      discretization = PhysicsInformedNN(chain,grid_strategy,param_estim = true, additional_loss = Hpen)
end

begin
      initθ = discretization.init_params
      acum =  [0;accumulate(+, length.(initθ))]
      sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
      phi = discretization.phi
end


begin
      addloss(phi,θ,p) = Hpen(phi,θ,p)
      prob = NeuralPDE.discretize(pde_system,discretization)

      res = GalacticOptim.solve(prob, BFGS();cb =cb,maxiters = 500)
      prob = remake(prob,u0 = res.minimizer)
      res = GalacticOptim.solve(prob, BFGS();cb =cb,maxiters = 500)
      #prob = remake(prob,u0 = res.minimizer)
      #res = GalacticOptim.solve(prob, ADAM(0.01);cb =cb,maxiters = 500)

      res.minimizer[end-1:end]
      initθ = discretization.init_params;
      acum =  [0;accumulate(+, length.(initθ))];
      sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1];
      phi = discretization.phi
      minimizers = [res.minimizer[s] for s in sep]
      u_predict = [[first(phi[i](t,minimizers[i])) for t in ts] for i in 1:2]
end

begin
      using Plots
      #
      # Analytical Solution for plotting

      hdata = [AnalyticalSol(a,-2.,10.)[4] for a in ts]
      udata =  [AnalyticalSol(a,-2.,10.)[5] for a in ts]
      vdata = [AnalyticalSol(a,-2.,10.)[3] for a in ts]
      u_ = [hdata'; vdata'; udata']
      t_ = hcat(collect(ts)...)# [2:end-1]
      len = length(t_)

      p11 = plot([u_[i,:] for i in 1:3],title = " exact solution ",yticks = -4.:1.:10)
      p12 = plot(ts, u_predict, label = ["h(t)" "v(t)" "u(t)"],xticks = 0.0 :0.5: 4.2,yticks = -4.:1.:10, title = "w/o data $(length(t_))pnts linear")
      plot(p11,p12)
end

#=
pnew= res.minimizer[end-1:end]
u3 =reduce(vcat,[pnew[1]*t+ pnew[2]*t^2 for t in t_])
#p13 = plot(ts,u3)

#=
# Function for TL and Hloss {{{
""" With additional loss function having summation of values """;
function TLandHloss(phi,θ)
 # Energy integral using Trapezoidal rule
      δt = ts[2] -ts[1]
      uₚ(i) = ([first(phi[i](t,θ[sep[i]])) for t in t_]) # θ = res.minimizer
      fg = abs.(reduce(vcat,uₚ(3)))
      up1 =  reduce(vcat,uₚ(1))
      sumofall = fg[1] + 2*sum(fg[2:end-1]) + fg[end] # sum of all the interior points in the grid and the exterior poitns are taken 0
      J1 = (δt/2)*sumofall

      # Penalty to ensure h > 0
      u21 =  [(up1[i] >=0.0) ? up1[i] : up1[i] .+ exp(-5*up1[i].*t2[i]) for i in 1:length(t_)]
      J2 = sum(u21)

      # Here W1, W2 are the weights
      W1 = 0.8; W2 = 1 - W1
      J = W1*J1 + W2*J2
      return J
end
function HpenIntsumofU(phi,θ,p)
      # creating penalty function for checking parameters
      #println("$(p)")
      uₚ(i) = ([first(phi[i](t,θ[sep[i]])) for t in t_]) # θ
      up1 =  reduce(vcat,uₚ(1))
      # Penalty to ensure h > 0
      u21 =  [(up1[i] >=0.0) ? up1[i] : up1[i]^2 for i in 1:length(t_)]
      J2 = sum(u21)

      # now integrated loss function of U should also be minimized so we use parameters from p and minimize U using it as below
      # we had  min ∫udt so, we define ∫ as ∑
      J1 = sum([p[1]*t+ p[2]*t^2 for t in t_])
      J1 = reduce(vcat,([pnew[1]*t+ pnew[2]*t^2 for t in t_]))

      # adding weights
      W1 = 0.8; W2 = 1-W1;
      J = W1*J1 + W2*J2
      return J
end
=#
