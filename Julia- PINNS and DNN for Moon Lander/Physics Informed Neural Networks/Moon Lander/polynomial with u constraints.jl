      """ POLYNOMIAL APPROXIMATION WITH  U CONSTRAINT AND ADDITIONAL LOSS FUNCTION IMPLEMENTED. """;
#u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] < 0.) ? upred[i].^2  : 3 for i in 1:length(t_)]

##
begin
      using NeuralPDE, ModelingToolkit, Flux, DiffEqFlux, GalacticOptim
      using Symbolics
      import ModelingToolkit: infimum, supremum, Interval
      using DomainSets, Quadrature
      using Plots
      using QuasiMonteCarlo, Distributions
      using Quadrature, Cubature
end


"""Since we cant solve ode and also estimate parameters just by giving tspan and initial conditions, let us
give the final conditions also and verify whether the paraemeters and minimum values obtained are constant""";
begin
      @parameters t,a,b,c,d,e
      @variables h(..),v(..), u(..)

      Dt = Differential(t)
      #  H, V are +ve upward direction , since flight is descending we have v0 = -2 as initial condition, and h0 =10.0 initially. As the aircraft reaches
      # the ground the Hfinal should be 0 and v final should reach 0. For this we need to minimize the energy consumption ∫Udt.

      # to solve this we are solving the system of ODE's and given the initial conditions. As a part of ModelingToolkit interface, we give the indvars and
      # depvars.

      """ parameters obtained for getting analytical udata are  a = 157, b = 221 """
      eqn = [Dt(h(t)) ~ v(t),
                  Dt(v(t)) ~ -1.5 + (a+b*t+c*t^2+d*t^4+e*t^7)]
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
      upred =   [(p[1]+p[2]*t+p[3]*t^2+p[4]*t^4+p[5]*t^7) for t in t_]
      u21 = upred # This will create  a same size vector of u21.
      """ Trapizoidal rule for loss""";
      u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] < 0.) ? upred[i].^2  : 3 for i in 1:length(t_)]
      sumofall = u21[1] + 2 * sum(u21[2:end-1]) + u21[end]

      J = abs((δt/2) * sumofall) * 1e-4

      return J
end

begin
      parameters = [a,b,c,d,e]
      pguess2 = [30,50,16,56,5]
      initGuess = Dict([p => pguess2[i] for (p,i) in zip(parameters,[1,2,3,4,5])])
      # Check to see if there is an ODE System function
      @named pde_system = PDESystem(eqn,bcs,domains,indvars,depvars,parameters,defaults = initGuess)
      n = 10
      chain = [FastChain(FastDense(1 ,n,σ),FastDense(n,n,σ),FastDense(n,1)) for _ in 1:2]
      dt = 0.01
      ts = 0.:dt:tf
      t_ = collect(ts)
      u21 = zeros(length(t_))
      grid_strategy = NeuralPDE.GridTraining(dt)
      discretization = PhysicsInformedNN(chain,grid_strategy,param_estim = true, additional_loss = ThrustLossfunc )
end

begin
      initθ = discretization.init_params
      acum =  [0;accumulate(+, length.(initθ))]
      sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
      phi = discretization.phi
end

begin
      prob = NeuralPDE.discretize(pde_system,discretization)
      res = GalacticOptim.solve(prob, BFGS();cb =cb,maxiters = 2500)
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
      p12 = plot(ts, u_predict, label = ["h(t)" "v(t)"],xticks = 0.0 :0.5: 4.2,yticks = -4.:1.:10, title = "NeuralPDE solution")
      println("parameters are:",res.minimizer[end-2:end])
      println("minimum value of h,v, vmax", minimum(u_predict[1]) ,  minimum(u_predict[2]), maximum(u_predict[2]))
      plot(p11,p12)
end
#png("D:\\Chaitanya\\sciml\\december\\20th december\\ grid with loss minimizing the objective function")

begin
      p = res.minimizer[end-4:end]
      #u21 =  [(upred[i] >=0.0 && upred[i] <= 3.) ? upred[i] : (upred[i] > 3.) ? 3 : upred[i].^2  for i in 1:length(t_)]

      Uvec =  [(p[1]+p[2]*t+p[3]*t^2+p[4]*t^4+p[5]*t^7) for t in t_]
      plot(title = "variation of predicted vs actual thrust")
      plot!(t_,udata, xticks = 0.0 :0.5:4.2, yticks = 0.0 :0.5:3.5, label ="Actual thrust" )
      plot!(t_,Uvec, xticks = 0.0 :0.5:4.2, yticks = 0.0 :0.5:3.5; label = "Thrust prediction")
end
#png("D:\\Chaitanya\\sciml\\december\\20th december\\ Thrust for optimal loss function")
println("BC init",[u_predict[1][1];u_predict[2][1];Uvec[1]])
println("params ",p)
println("BC end ",[u_predict[1][end]; u_predict[2][end] ;Uvec[end]])
