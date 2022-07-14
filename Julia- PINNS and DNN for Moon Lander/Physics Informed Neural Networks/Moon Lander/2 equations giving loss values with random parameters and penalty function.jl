begin
      using NeuralPDE, ModelingToolkit, Flux, DiffEqFlux, GalacticOptim
      using Symbolics
      import ModelingToolkit: infimum, supremum, Interval
      using DomainSets, Quadrature
      using Plots
      using QuasiMonteCarlo, Distributions
      using Quadrature, Cubature
end


"""Since we cant solve ode and also estimate parameters1 just by giving tspan and initial conditions, let us
give the final conditions also and verify whether the paraemeters and minimum values obtained are constant""";
begin
      @parameters1 t,a,b,c
      @variables h(..),v(..), u(..)

      Dt = Differential(t)
      #  H, V are +ve upward direction , since flight is descending we have v0 = -2 as initial condition, and h0 =10.0 initially. As the aircraft reaches
      # the ground the Hfinal should be 0 and v final should reach 0. For this we need to minimize the energy consumption ∫Udt.

      # to solve this we are solving the system of ODE's and given the initial conditions. As a part of ModelingToolkit interface, we give the indvars and
      # depvars.

      """ parameters1 obtained for getting analytical udata are  a = 157, b = 221 """
      eqn = [Dt(h(t)) ~ v(t),
                  Dt(v(t)) ~ -1.5 + c/(1+exp(-a*t+b))]
      v0= -2.; h0 = 10.
      tf = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))

      begin
            bcs = [h(0.) ~ 10.,
                    v(0.) ~ -2., # its decelerating
                    h(tf) ~ 0.,
                    v(tf) ~ 0.]
      end

      domains = [t in Interval(0.,tf)]
      indvars = [t]
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
      """ We wanted to reduce the Thrust value  i.e  J:= ∫udt so for U we need to get the parameters1 of a,b from each
      iteration in the res.minimizer""";
end

function ThrustLossfunc(phi,θ,p)
      δt = dt                 # for trapizoidal loss function we need this
      # since we assumed linear combination of U from 0 < U < 3 we find the values w.r.t params obtained by training
      upred =  [p[3]/(1+exp(-p[1]*t+p[2])) for t in t_]
      u21 = upred
      """ Trapizoidal rule for loss""";
      sumofall = u21[1] + 2 * sum(u21[2:end-1]) + u21[end]
      J = (δt/2) * sumofall *1e-2
      return J
end
#pguess = 5
#initGuess = Dict([p => pguess for p in parameters1])
begin
      parameters1 = [a,b,c]
      pguess2 = [150,220,3]

      initGuess = Dict([p => pguess2[i] for (p,i) in zip(parameters1,[1,2,3])])
      # Check to see if there is an ODE System function
      @named pde_system = PDESystem(eqn,bcs,domains,indvars,depvars,parameters1,defaults = initGuess)
      n = 10
      chain = [FastChain(FastDense(1 ,n,σ),FastDense(n,n,σ),FastDense(n,1)) for _ in 1:3]

      points = 100
      ts = LinRange(0.,tf,points)
      dt = ts[2]-ts[1]
      t_ = collect(ts)
      u21 = zeros(length(t_))
      quasiRand_strategy =  NeuralPDE.QuasiRandomTraining(points ;bcs_points = points, sampling_alg = LatinHypercubeSample(),resampling =true, minibatch=0)
      grid_strategy = NeuralPDE.GridTraining(dt);
      discretization = PhysicsInformedNN(chain,quasiRand_strategy,param_estim = true, additional_loss = nothing)
      #discretization = PhysicsInformedNN(chain,grid_strategy,param_estim = true, additional_loss = ThrustLossfunc)

end

begin
      initθ = discretization.init_params
      acum =  [0;accumulate(+, length.(initθ))]
      sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
      phi = discretization.phi
end


begin
      prob = NeuralPDE.discretize(pde_system,discretization)
      res = GalacticOptim.solve(prob, BFGS();cb =cb,maxiters = 3000)
      initθ = discretization.init_params;
      acum =  [0;accumulate(+, length.(initθ))];
      sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1];
      phi = discretization.phi
      minimizers = [res.minimizer[s] for s in sep]
      u_predict = [[first(phi[i](t,minimizers[i])) for t in ts] for i in 1:2]
end

begin
      using Plots
      hdata = [AnalyticalSol(a,-2.,10.)[4] for a in ts]
      udata =  [AnalyticalSol(a,-2.,10.)[5] for a in ts]
      vdata = [AnalyticalSol(a,-2.,10.)[3] for a in ts]
      u_ = [hdata'; vdata'; udata']
      len = length(t_)
      p11 = plot(ts,[u_[i,:] for i in 1:3],title = " exact solution ",yticks = -4.:1.:10,xticks = 0.0 :0.5: 4.2)
      p12 = plot(ts, u_predict, label = ["h(t)" "v(t)" "u(t)"],xticks = 0.0 :0.5: 4.2,yticks = -4.:1.:10, title = "NeuralPDE solution
      with initial guess
      $(pguess)")
      println("parameters1 are:",res.minimizer[end-2:end])
      println("minimum value of h,v, vmax", minimum(u_predict[1]) ,  minimum(u_predict[2]), maximum(u_predict[2]))
      plot(p11,p12)
end
#png("D:\\Chaitanya\\sciml\\december\\20-12-2021 quasirand with p 50 and random parameter")


## Not used now
p = res.minimizer[end-2:end]
Uvec =  [p[3]/(1+exp(-p[1]*t+p[2])) for t in t_]
plot(title = "vairation of predicted vs actual thrust")
plot!(t_,udata, xticks = 0.0 :0.5:4.2, yticks = 0.0 :0.5:3.5, label ="Actual thrust" )
plot!(t_,Uvec, xticks = 0.0 :0.5:4.2, yticks = 0.0 :0.5:3.5; label = "Thrust prediction")
#png("D:\\Chaitanya\\sciml\\december\\20-12-2021 upredict vs uactual with random parameter")


"""
Q1) why sigmoid?
because we dont know the type of solution we obtain we chose a nonlinear function i.e sigmoid, we could also choose tanh

2) neural pde is able to capture the values based on the bc and obtain the constant that is necessary to be multipled with
the nonlinear function.

3) need to check different training strategies.


Q4) what if we dont know the final time? how do we calculate it using the conditions itself?
Ans) we should know at least 1 condition to work with, here we had time as constraint and within which we were able to generate the  solution.

Q5) Implement chebyshev polynomials and for different trajectory optimization problem.

Q6)
""";
