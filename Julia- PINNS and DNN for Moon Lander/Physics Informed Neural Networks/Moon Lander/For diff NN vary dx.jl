""" For the same neural netwrok with same initial parameters when we change grid size we obtain same trends""";
##
""" Packages usage """;
begin
        using NeuralPDE, ModelingToolkit, DiffEqFlux, GalacticOptim, Plots
        using DiffEqOperators
        using DifferentialEquations
        using DomainSets
        using OrdinaryDiffEq
        using CUDA
end
##
""" Symbolic representation""";
gridsizes = [0.1 0.01 0.001]  ;     # grid training;

begin
        plot1 = plot()
        for dx in gridsizes
                begin
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
                end

                begin                 """ Defining NeuralNetwork """;;

                        Hu = 16 #hiddenunits =16;
                        layers = 2;
                        chain = FastChain(FastDense(2,Hu ,σ),FastDense(Hu ,Hu ,σ),FastDense(Hu,1,σ));
                        initθ = Float64.(DiffEqFlux.initial_params(chain)) |>gpu;
                        indvars = [t,x];
                        depvars = [u(t,x)];
                end
                ##
                """ Discretize and Optimize""";
                begin
                        discretization = PhysicsInformedNN(chain,GridTraining(dx);inital_params = initθ);
                        @named pdesys = PDESystem(eqn,bcs,domains,indvars,depvars);
                        prob = discretize(pdesys,discretization);
                        cb_ = function (p,l)
                            #println("loss: ", l , "losses: ", map(l -> l(p), lossfunctions))
                            println("the current loss is : $l")
                            return false
                        end;
                        opt1 = BFGS();opt2 = ADAM(0.01);
                        maxiters=50;
                        res = GalacticOptim.solve(prob, opt2; cb = cb_, maxiters=maxiters)
                        phi = discretization.phi
                end
                ##
                ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]

                """ [[phi([t,x],res.minimizer) for x in xs] for t in ts][1]
                This statement indicates for t in each ts we will calculate all points of x """;

                u_predict = hcat([[first(phi([t,x],res.minimizer)) for x in xs] for t in ts]...)
                u_ex = (t,x) -> exp.(-t) * cos.(x);
                u_exact = [u_ex(t,x) for x in xs , t ∈ ts]

                function norms(ts,xs,f,g)
                         l2 = sum([((f .- g).^2)[:,i] for i ∈ 1: length(ts)])./(length(xs))
                         l2percent = sum([(((u_predict.-u_exact)./u_exact).^2)[:,i] for i in 1 :length(xs)])
                         l∞ = maximum([(abs.(f .- g))[:,i] for i in 1: length(ts)])
                         return l2, l∞
                end
                l2norm,linfinity = norms(ts,xs,u_predict,u_exact)


        """ LOG PLOTS""";
        theme(:bright)
        plot!(ts,log.(l2norm),title = "L2 Norm for different
       sizes of δx", xticks = 0.0:0.25:1.0,
                      xlabel = "time in (s)",ylabel = "log(ϵ)", xlims = (0.0,1.0), ylims = (-16.0,0.0),
                      yticks = 0.0:-1.0:-16.0, frame =:box,label = "at δx = $(dx)")
    end
    display(plot1)
   # png("D:l2norm for various dx vals")

end
png("D:\\Chaitanya\\sciml\\Analysis so far\\ for different grid and rerun the problem")
