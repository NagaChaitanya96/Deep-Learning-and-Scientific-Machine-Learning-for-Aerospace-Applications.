begin
      using NeuralPDE, ModelingToolkit, Flux, DiffEqFlux, GalacticOptim
      using Symbolics
      import ModelingToolkit: infimum, supremum, Interval
      using DomainSets, Quadrature
      using Plots
      using QuasiMonteCarlo, Distributions
      using Quadrature, Cubature
end

@parameters t,a,b
@variables h(..),v(..), u(..)
#Iₜ = Symbolics.Integral(t in ClosedInterval(0,tf))

Dt = Differential(t)

eqn = [Dt(h(t)) ~ v(t),
          Dt(v(t)) ~ -1.5 + (a+b*t),
          Dt(u(t)) ~ b]

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

begin
      domains = [t in Interval(0.,tf)]
      input_ = 1
      chain = [FastChain(FastDense(input_,10,σ),FastDense(10,10,σ),FastDense(10,1)) for _ in 1:3]
      initθ = map(c1 -> Float64.(DiffEqFlux.initial_params(c1)),chain)
      flat_initθ = reduce(vcat,initθ)
      dt = 0.05
      ts = 0. : dt : tf
      grid_strategy = NeuralPDE.GridTraining(dt)
      points = 100
      ts = LinRange(0.,tf,points)
      quasiRand_strategy =  NeuralPDE.QuasiRandomTraining(points ;bcs_points = points, sampling_alg = LatinHypercubeSample(),resampling =true, minibatch=0)
      Quadrature_strategy = NeuralPDE.QuadratureTraining(;quadrature_alg = CubatureJLh(), reltol =1e-6, abstol =1e-3, maxiters =1e3, batch = 100)
      indvars = [t]
      depvars =[h,v,u]
      ps = [a,b]
      defaults = Dict([p => 0.1 for p in ps])
end

begin
      function AnalyticalSol(t,v0,h0)
            tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
            ŝ = tf̂/2 + v0/3
            v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
            ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
            û(t) = (t<=ŝ) ? 0 : 3
            #println("The values are : $([tf̂, ŝ ,v̂(t) , ĥ(t), û(t)])")
            return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
      end


      hdata = [AnalyticalSol(a,-2.,10.)[4] for a in ts]
      udata =  [AnalyticalSol(a,-2.,10.)[5] for a in ts]
      vdata = [AnalyticalSol(a,-2.,10.)[3] for a in ts]
      u_ = [hdata'; vdata'; udata']
      t_ = hcat(collect(ts)...)# [2:end-1]
      len = length(t_)

end


""" With additional loss function having summation of values """;
u21 = zeros(length(t_))
begin
      t2 = reduce(vcat,t_)
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

      #=function TLandHloss(phi,θ,p)
        # Energy integral using Trapezoidal rule
            δt = ts[2] -ts[1]
            uₚ(i) = ([first(phi[i](t,θ[sep[i]])) for t in t_]) # θ = res.minimizer

            #fg = abs.(reduce(vcat,uₚ(3)))
            up1 =  reduce(vcat,uₚ(1))

            fg = [p[1]*t+p[2]*t^2 for t in t_]
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
      =#
      function Hpen(phi,θ,p)
            println("$(p)")
            uₚ(i) = ([first(phi[i](t,θ[sep[i]])) for t in t_]) # θ
            up1 =  reduce(vcat,uₚ(1))
            # Penalty to ensure h > 0
            #u21 = up1
            #u21 =  [(up1[i] >=0.0) ? up1[i] : up1[i] .+ exp(-2*up1[i].*t2[i]) for i in 1:length(t_)]
            u21 =  [(up1[i] >=0.0) ? up1[i] : up1[i].^2 for i in 1:length(t_)]

            J2 = sum(u21)
            return J2
      end
      #Hpen = Hpenality(phi,θ,[]) # here if the neural network parameters are unfeasible we will be getting the H < 0. So
      # we added penalty and the sum of the penalty is assumed to reach 0.
      #addloss(phi,θ,p) = 0.1*Hpenality(phi,θ)+TL(phi,θ)
      addloss(phi,θ,p) =TLandHloss(phi,θ)
      discretization = PhysicsInformedNN(chain,grid_strategy,param_estim = true ,additional_loss = addloss)

      @named pde_system = PDESystem(eqn,bcs,domains,indvars,[h(t),v(t),u(t)],ps,defaults =defaults)
      prob = discretize(pde_system,discretization)
      cb = function(p,l)
              println("The loss is :",l)
              return false
      end
end

begin
      initθ = discretization.init_params
      acum =  [0;accumulate(+, length.(initθ))]
      sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
      phi = discretization.phi
end

# here if the neural network parameters are unfeasible we will be getting the H < 0. So

begin
      res = GalacticOptim.solve(prob, BFGS();cb =cb,maxiters = 50)
      prob = remake(prob,u0 = res.minimizer)
      res = GalacticOptim.solve(prob, ADAM(0.01);cb =cb,maxiters = 1000)

      minimizers = [res.minimizer[s] for s in sep]
      u_predict = [[first(phi[i](t,minimizers[i])) for t in ts] for i in 1:3]
      p11 = plot([u_[i,:] for i in 1:3],title = " exact solution ",yticks = -4.:1.:10)
      p12 = plot(ts, u_predict, label = ["h(t)" "v(t)" "u(t)"],xticks = 0.0 :0.5: 4.2,yticks = -4.:1.:10, title = "w/o data $(points)pnts O(n^5)")
      plot(p11,p12)
end
