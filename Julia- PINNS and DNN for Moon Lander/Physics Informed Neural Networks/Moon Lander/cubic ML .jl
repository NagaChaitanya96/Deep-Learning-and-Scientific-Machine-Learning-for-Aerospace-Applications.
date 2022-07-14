begin
      using ModelingToolkit
      using DiffEqFlux
      using NeuralPDE
      import ModelingToolkit: Interval
      using GalacticOptim
end


"""Since we cant solve ode and also estimate parameters just by giving tspan and initial conditions, let us
give the final conditions also and verify whether the paraemeters and minimum values obtained are constant""";
begin
      @parameters t,a,b,c,d
      @variables h(..),v(..), u(..)

      Dt = Differential(t)
      #  H, V are +ve upward direction , since flight is descending we have v0 = -2 as initial condition, and h0 =10.0 initially. As the aircraft reaches
      # the ground the Hfinal should be 0 and v final should reach 0. For this we need to minimize the energy consumption ∫Udt.

      # to solve this we are solving the system of ODE's and given the initial conditions. As a part of ModelingToolkit interface, we give the indvars and
      # depvars.
      eqn = [Dt(h(t)) ~ v(t),
                Dt(v(t)) ~ -1.5 + (a+b*t + c*t^2 + d*t^3)]

      v0= -2.; h0 = 10.
      tf = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))

      bcs = [h(0.) ~ h0,
             v(0.) ~ v0,# its deceleratin
             h(tf) ~ 0.,
             v(tf) ~ 0.]

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
      """ We wanted to reduce the Thrust value  i.e  J:= ∫udt so for U we need to get the parameters of a,b from each
      iteration in the res.minimizer""";
end

function ThrustLossfunc(phi,θ,p)
      δt = dt                 # for trapizoidal loss function we need this
      # since we assumed linear combination of U from 0 < U < 3 we find the values w.r.t params obtained by training
      upred = [p[1] + p[2]*t + p[3]*t^2 + p[4] *t^3 for t in t_]

      """ control path constraint 0 < u < 3 """;
      #u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] > 3.) ? 3 : upred[i].^2  for i in 1:length(t_)]
      u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] > 3.) ? 3 : 0  for i in 1:length(t_)]

      """ Trapizoidal rule for loss""";
      sumofall = u21[1] + 2 * sum(u21[2:end-1]) + u21[end]
      J = (δt/2) * sumofall * 0.05
      return J
end

begin
      parameters = [a,b,c,d]
      initGuess = Dict([p => 1. for p in parameters])
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
      discretization = PhysicsInformedNN(chain,grid_strategy,param_estim = true, additional_loss = nothing)
      #discretization = PhysicsInformedNN(chain,grid_strategy,param_estim = true, additional_loss = ThrustLossfunc)
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

      res = GalacticOptim.solve(prob, BFGS();cb =cb,maxiters = 1500)
      prob = remake(prob,u0 = res.minimizer)
      res = GalacticOptim.solve(prob, BFGS();cb =cb,maxiters = 1500)
      prob = remake(prob,u0 = res.minimizer)
      res = GalacticOptim.solve(prob, ADAM(0.01);cb =cb,maxiters = 1500)

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
      len = length(t_)

      p11 = plot([u_[i,:] for i in 1:3],title = " exact solution ",yticks = -4.:1.:10)
      p12 = plot(ts, u_predict, label = ["h(t)" "v(t)" "u(t)"],xticks = 0.0 :0.5: 4.2,yticks = -4.:1.:10, title = "w/o data $(length(t_))pnts linear")
      #println("parameters are:",res.minimizer[end-1:end])
      println("parameters are:",res.minimizer[end-3:end])
      println("minimum value of h,v, vmax", minimum(u_predict[1]) ,  minimum(u_predict[2]), maximum(u_predict[2]))
      plot(p11,p12)

end

""" Approximating U values from the ḣ and v̇ equation """;

p = res.minimizer[end-3:end]
Uvec = [p[1] + p[2]*t + p[3]*t^2 + p[4] *t^3 for t in t_]
plot(t_,Uvec)


#=
u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] > 3.) ? 3 : 0  for i in 1:length(t_)]
#u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] > 3.) ? 3 : upred[i].^2  for i in 1:length(t_)]

sumofall = u21[1] + 2 * sum(u21[2:end-1]) + u21[end]
J = (dt/2) * sumofall *0.05
