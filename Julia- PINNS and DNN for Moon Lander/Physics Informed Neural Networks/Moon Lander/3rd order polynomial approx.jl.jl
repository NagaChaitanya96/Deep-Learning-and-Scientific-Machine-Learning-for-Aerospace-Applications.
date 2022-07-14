begin
      using NeuralPDE, ModelingToolkit, Flux, DiffEqFlux, GalacticOptim
      using Symbolics
      import ModelingToolkit: infimum, supremum, Interval
      using DomainSets, Quadrature
      using Plots
      using QuasiMonteCarlo, Distributions
      using Quadrature, Cubature
end
@parameters t,a,b,c,d
@variables h(..),v(..), u(..)

Dt = Differential(t)

"""
Q) should we need to give du/dt as a neural network? and if so why?
ans)                                 points to be noted:

      1. We have only 2 equations ḣ and v̇ and the boundary conditions for them are also known.
      2. ḣ = f(v); v̇ = f(u); which is unknown. So we are experimenting u as a linear model to a non linear model
         which could fit the corresponding boundary conditions.
      3. suppose we took a linear model u(t) = a + bt; then we need to find params of a,b which could satisfy the
         defined equations ḣ and v̇. But we also have the control constraints such as 0 <= u <= 3.

Q2) Again a question araises how do we add constraints in our model? when we dont define u as a seperate neural network?
ans) we tried adding the penality loss function that could always make the u constraint > 0 and < 3.
      problem with this model is,
       1. {
                  since our model has  a loss function which is sum of the PDE + BC loss function, we need to precisely know
                  how much effect our BC and PDE losses are effecting the solution parameters. Suppose if we add another
                  loss function such as additional loss, it may happen that; the effect of additional loss is >> combined
                  PDE +BC loss; which makes our optimizer to only minimize the additional loss function, i.e. the constraint
                  0 < u < 3.

                  example : eqn = [Dt(h(t)) ~ v(t),
                            Dt(v(t)) ~ -1.5 + (a+b*t),

                        {Our assumption is u = a+ b*t }

                  Boundary:  bcs = [h(0.) ~ 10.,
                           v(0.) ~ -2.,
                           h(tf) ~ 0.,
                           v(tf) ~ 0.]

                  additional loss = {
                        upred = [p[1]+ p[2]*t for t in t_ ]
                        {
                            Imposing Hard constraint
                            u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] > 3.) ? 3 : upred[i].^2  for i in 1:length(t_)]
                        }
                        {
                            Imposing soft constraint which is chosen based on intuition

                            if U > 3
                                  U - log(U)
                            elseif U < 0
                                  U+ exp(-U)
                            elseif U >0. && U <3.
                                  U
                            end

                            u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] > 3.) ? upred[i] - log(upred[i]) : upred[i] + exp(-upred[i])  for i in 1:length(t_)]
                            sumofall = u21[1] + 2 * sum(u21[2:end-1]) + u21[end]
                        }


                        we can give a vector of penality and try to run the code for different penalities from 0.0001 to 0.005
                        penalty = 0.005
                        J = (δt/2) * sumofall * penalty
                        }

                  So we need to randomly keep on trying different weights for the additional loss function until we get the
                  required analytical result.
                  However, in real life we dont know the actual trajectory of the system, so we cant completely relay
                  on trail and error base of solution.

       }

       2. {
            One more way to do this is to assume du/dt as a neural network and approximate it to a function of our choice
            and impose the constraints u(0) and u(tf) as BC's. and let the model train the parameters w.r.t the BC's such as
            eqn = [Dt(h(t)) ~ v(t),
                      Dt(v(t)) ~ -1.5 + (a+b*t),
                      Dt(u(t)) ~ b]

                      bcs = [h(0.) ~ 10.,
                              v(0.) ~ -2., # its decelerating
                              u(0.) ~ 0.,
                              u(tf) ~ 3.,
                              h(tf) ~ 0.,
                              v(tf) ~ 0.,
                              ]

            But the caveats with this method is, NN will train params a,b w.r.t the approximated equation on the RHS. So
            if we have an initial idea of how our Trajectory of the Model looks like we can build an approximate RHS function
            and train the NN(θ,λ) and hyperparameters a,b respectively.

      3. {
            So far we have assumed that the final trajectory time is known, however, the time and path of actual case will be
            unknown.
            }

       }

       4. Instead of approximating RHS of a Neural Network as a polynomial function or Chebeshev polynomials, can we approximate
          it as another NN? if so how do we give a neural network in ModelingToolkit ?

""";

#=
eqn = [Dt(h(t)) ~ v(t),
          Dt(v(t)) ~ -1.5 + (a+b*t),
          Dt(u(t)) ~ b]
=#


eqn = [Dt(h(t)) ~ v(t),
                    Dt(v(t)) ~ -1.5 + (a+b*t+c*t^2+d*t^3),
                    Dt(u(t)) ~ b+2*c*t+3*d*t^2]
begin
      v0= -2.; h0 = 10.
      tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
      tf = tf̂
end
begin
      bcs = [h(0.) ~ 10.,
              v(0.) ~ -2., # its decelerating
              u(0.) ~ 0.,
              u(tf) ~ 3.,
              h(tf) ~ 0.,
              v(tf) ~ 0.,
              ]
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
      upred = [1/(1+exp(-(p[1]+p[2]*t))) for t in t_]

      """ control path constraint 0 < u < 3 """;
      u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] > 3.) ? 3 : upred[i].^2  for i in 1:length(t_)]
      """ Trapizoidal rule for loss""";
      sumofall = u21[1] + 2 * sum(u21[2:end-1]) + u21[end]
      J = (δt/2) * sumofall *0.005

      return J
end

begin
      n = 10
      chain = [FastChain(FastDense(1 ,n,σ),FastDense(n,n,σ),FastDense(n,1)) for _ in 1:3]
      parameters = [a,b,c,d]
      initGuess = Dict([p => 1. for p in parameters])
      indvars = [t]
      depvars = [h(t),v(t),u(t)]
      # Check to see if there is an ODE System function
      @named pde_system = PDESystem(eqn,bcs,domains,indvars,depvars,parameters,defaults = initGuess)
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

      res.minimizer[end-1:end]
      initθ = discretization.init_params;
      acum =  [0;accumulate(+, length.(initθ))];
      sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1];
      phi = discretization.phi
      minimizers = [res.minimizer[s] for s in sep]
      u_predict = [[first(phi[i](t,minimizers[i])) for t in ts] for i in 1:3]
end

begin
      using Plots
      # Analytical Solution for plotting

      hdata = [AnalyticalSol(a,-2.,10.)[4] for a in ts]
      udata =  [AnalyticalSol(a,-2.,10.)[5] for a in ts]
      vdata = [AnalyticalSol(a,-2.,10.)[3] for a in ts]
      u_ = [hdata'; vdata'; udata']
      len = length(t_)

      p11 = plot([u_[i,:] for i in 1:3],title = " exact solution ",yticks = -4.:1.:10)
      p12 = plot(ts, u_predict, label = ["h(t)" "v(t)" "u(t)"],xticks = 0.0 :0.5: 4.2,yticks = -4.:1.:10, title = "w/o data $(length(t_))pnts linear")
      #println("parameters are:",res.minimizer[end-1:end])
      println("parameters are:",res.minimizer[end-1:end])
      println("minimum value of h,v, vmax", minimum(u_predict[1]) ,  minimum(u_predict[2]), maximum(u_predict[2]))
      plot(p11,p12)

end


""" Polynomial cannot fit a step function.""";
