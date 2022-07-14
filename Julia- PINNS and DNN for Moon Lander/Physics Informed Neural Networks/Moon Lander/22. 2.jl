begin
    using NeuralPDE, ModelingToolkit, DiffEqFlux, GalacticOptim, Plots
    using DifferentialEquations
    using DiffEqOperators
    using DomainSets
    using OrdinaryDiffEq
    using CUDA
    using BenchmarkTools
end
##
@btime begin

    begin
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
        dx = 0.1       # grid training;
    end

    ##
    begin
        """ Defining NeuralNetwork """;;
        Hu =16 #hiddenunits =16;
        layers = 3;

        chain = FastChain(FastDense(2,Hu ,σ),FastDense(Hu ,Hu ,σ),FastDense(Hu ,Hu ,σ),FastDense(Hu,1,σ));
        initθ = Float64.(DiffEqFlux.initial_params(chain)) |>gpu;
        discretization = PhysicsInformedNN(chain,GridTraining(dx);inital_params = initθ);
        indvars = [t,x];
        depvars = [u(t,x)];
        @named pdesys = PDESystem(eqn,bcs,domains,indvars,depvars);
        prob = discretize(pdesys,discretization);
    end


    cb_ = function (p,l)
        #println("loss: ", l , "losses: ", map(l -> l(p), lossfunctions))
        println("the current loss is : $l")
        return false
    end;

    begin
        """ solving the optimization problem""";
        opt1 = BFGS(); opt2 = ADAM(0.01);
        maxiters=25;
        res = GalacticOptim.solve(prob, opt1; cb = cb_, maxiters=maxiters);
        prob = remake(prob,u0= res.minimizer)
        res = GalacticOptim.solve(prob, opt2; cb = cb_, maxiters=maxiters);
        phi = discretization.phi
    end
    ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
    u_ex = (t,x) -> exp.(-t) * cos.(x);
    u_predict = hcat([[first(phi([t,x],res.minimizer)) for x in xs] for t in ts]...)


    begin  """ solutions """;
        u_predict = reshape([first(phi([t,x],res.minimizer)) for t in ts for x in xs ],(length(xs),length(ts)))
        u_exact = reshape([u_ex(t,x) for t in ts for x in xs], (length(xs),length(ts)))
    end

    begin
            l2norm = sum([((u_predict .- u_exact).^2)[:,i] for i ∈ 1: size(u_predict)[1]])./(size(u_predict))[2]
            plot(xs,log10.(l2norm),title = "L2 norm using NPDE",label ="BFGS() + ADAM(0.01) with
            each Iter =$(maxiters)" , xticks = 0.0:0.25:1.0)
            xlabel!("Distance X")
            ylabel!("log10(Error)")
            #png("D:\\correct a l2 NPDE")

            linfinity = maximum([(abs.(u_predict .- u_exact))[:,i] for i in size(u_predict)[1]])./size(u_predict)[2]
            plot(xs,log10.(linfinity),title = "L∞ norm using NPDE",xticks = 0.0:0.25:1.0)
            xlabel!("Distance X")
            ylabel!("log10(Error) ")
            #png("D:\\correct a l∞ NPDE")
    end
end
