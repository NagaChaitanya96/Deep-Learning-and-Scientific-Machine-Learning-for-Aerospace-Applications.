""" For the same amount of NN architecture varying the robustness from 1 : 10 % """;
cd(@__DIR__)
begin
    using NeuralPDE, ModelingToolkit, Flux, DiffEqFlux, GalacticOptim
    using Symbolics
    import ModelingToolkit: infimum, supremum, Interval
    using DomainSets, Quadrature
    using Plots
    using QuasiMonteCarlo, Distributions
    using Quadrature, Cubature
    using BenchmarkTools
end

#include("State_variation.jl")
#include("NN_architecture.jl")

""" Analytical solution details""";
v0= -2.; h0 = 10.

tfinal_(h0,v0) = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
tf = tfinal_(h0,v0)
function AnalyticalSol(t,v0,h0)
    tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
    ŝ = tf̂/2 + v0/3
    v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
    ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
    û(t) = (t<=ŝ) ? 0 : 3
    return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
end

##
""" Functions """;
using Random
n1 = [1 5 10];
dists = [(-a:1:a)*0.01 for a in n1];
num = [100 500 1000];
x1,y1,xt,yt =[],[],[],[];
idx=[];
""" Generating test data for error approximation """;
htest,vtest = 10.015,-2.065 #vec([maximum(h_dist)+0.0015*a for a in [1 5 10]]) , vec([minimum(v_dist)-0.0015*a for a in [1 5 10]]);
step2 = 100
tftest = tfinal_(htest,vtest)
htd = [AnalyticalSol(b,vtest,htest)[4] for b in LinRange(0,tftest,step2)]
vtd = [AnalyticalSol(b,vtest,htest)[3] for b in LinRange(0,tftest,step2)]
utd = [AnalyticalSol(b,vtest,htest)[5] for b in LinRange(0,tftest,step2)]
all_data = [];
test_all =[];
utd1 =[];
err_train= [];
err_test= [];
model_vec = [];
iter = 5;
η = 0.005; n = 16;
##
step1 = num[1]
a = n[1]


dist = (-1:0.2:1)*0.01*a
global v_dist,h_dist = v0.+collect(dist), h0.+collect(dist)
tf_dist = [(2/3)*(v1)+ (4/3*(sqrt((v1^2/2) + 1.5*h1))) for v1 in v_dist, h1 in h_dist]
h_matrix = [];
u_matrix = [];
v_matrix = [];
global h_mat,v_mat,u_mat,t_all
h_mat = [];
v_mat = [];
u_mat = [];
t_all = [];

for v0 in v_dist
    for h0 in h_dist
        tfinal = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
        h_data = [AnalyticalSol(a,v0,h0)[4] for a in LinRange(0,tfinal,step1)]
        v_data = [AnalyticalSol(a,v0,h0)[3] for a in LinRange(0,tfinal,step1)]
        u_data = [AnalyticalSol(a,v0,h0)[5] for a in LinRange(0,tfinal,step1)]

        h_matrix = push!(h_matrix,h_data)
        v_matrix = push!(v_matrix,v_data)
        u_matrix = push!(u_matrix,u_data)

        h_mat = hcat(h_matrix...)
        v_mat = hcat(v_matrix...)
        u_mat = hcat(u_matrix...)
        t_all = push!(t_all,tfinal)
    end
end

#t_all == vec(tf_dist') # this states that for each combination of v0,h0 we have enough data and that is true.
h_data  = vec(h_mat)
v_data  = vec(v_mat)
u_data  = vec(u_mat)
ip = [h_data' ; v_data']
op = hcat(u_data...)

""" DONT USE SHUFFLE , BECAUSE IT SHUFFLES ALL THE DATA IN ENTIRE MATRIX.""";
ss = Int64.(0.80*length(u_data))
rng = MersenneTwister(1234)

""" Shuffling done here""";

numbers = shuffle(rng,Vector(1:length(u_data)))
data2 = [h_data v_data u_data]
data2 = [data2[i,:] for i in eachrow(numbers)]
data2 = reduce(vcat,data2)

""" Setting to  test and train sets """;

X = data2[:,[1,2]]'
Y = data2[:,3]'
global x11 , y11 = X[:,1:ss] , Y[:,1:ss]
global xt1, yt1 = X[:,ss+1:end] , Y[:,ss+1: end]
x1,y1,xt,yt  = push!(x1,x11),push!(y1,y11),push!(xt,xt1),push!(yt,yt1)
idx1 = (step1,a)
push!(idx,idx1)


""" Defining a neural network""";
global model = Chain(Dense(2,n,σ),Dense(n,n,leakyrelu),Dense(n,n,relu),Dense(n,1,σ),x->3*x)
function loss(a,b)
    Flux.Losses.mse(model(a),b)
end
opt = ADAM(η);
data = Iterators.repeated((x11,y11),iter)
ps = Flux.params(model)

function cb!()
    push!(err_train,loss(x11,y11))
    push!(err_test, loss(xt1,yt1))
    println("loss of the function is :" ,loss(x11,y11))
end

Flux.train!(loss,Flux.params(model),data,opt,cb = cb!)
pred_vec = vec(model([htd';vtd']))
push!(all_data,pred_vec)
# saving all the trained models having same architecture, Learningrate, etc, but different
# training set of data
push!(model_vec,model)

##
for step1 in num[1]
    for a in n1[1]
        dist = (-1:0.2:1)*0.01*a
        global v_dist,h_dist = v0.+collect(dist), h0.+collect(dist)
        tf_dist = [(2/3)*(v1)+ (4/3*(sqrt((v1^2/2) + 1.5*h1))) for v1 in v_dist, h1 in h_dist]
        h_matrix = [];
        u_matrix = [];
        v_matrix = [];
        global h_mat,v_mat,u_mat,t_all
        h_mat = [];
        v_mat = [];
        u_mat = [];
        t_all = [];

        for v0 in v_dist
            for h0 in h_dist
                tfinal = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
                h_data = [AnalyticalSol(a,v0,h0)[4] for a in LinRange(0,tfinal,step1)]
                v_data = [AnalyticalSol(a,v0,h0)[3] for a in LinRange(0,tfinal,step1)]
                u_data = [AnalyticalSol(a,v0,h0)[5] for a in LinRange(0,tfinal,step1)]

                h_matrix = push!(h_matrix,h_data)
                v_matrix = push!(v_matrix,v_data)
                u_matrix = push!(u_matrix,u_data)

                h_mat = hcat(h_matrix...)
                v_mat = hcat(v_matrix...)
                u_mat = hcat(u_matrix...)
                t_all = push!(t_all,tfinal)
            end
        end

        #t_all == vec(tf_dist') # this states that for each combination of v0,h0 we have enough data and that is true.
        h_data  = vec(h_mat)
        v_data  = vec(v_mat)
        u_data  = vec(u_mat)
        ip = [h_data' ; v_data']
        op = hcat(u_data...)

        """ DONT USE SHUFFLE , BECAUSE IT SHUFFLES ALL THE DATA IN ENTIRE MATRIX.""";
        ss = Int64.(0.80*length(u_data))
        rng = MersenneTwister(1234)

        """ Shuffling done here""";

        numbers = shuffle(rng,Vector(1:length(u_data)))
        data2 = [h_data v_data u_data]
        data2 = [data2[i,:] for i in eachrow(numbers)]
        data2 = reduce(vcat,data2)

        """ Setting to  test and train sets """;

        X = data2[:,[1,2]]'
        Y = data2[:,3]'
        global x11 , y11 = X[:,1:ss] , Y[:,1:ss]
        global xt1, yt1 = X[:,ss+1:end] , Y[:,ss+1: end]
        x1,y1,xt,yt  = push!(x1,x11),push!(y1,y11),push!(xt,xt1),push!(yt,yt1)
        idx1 = (step1,a)
        push!(idx,idx1)


        """ Defining a neural network""";
        global model = Chain(Dense(2,n,σ),Dense(n,n,leakyrelu),Dense(n,n,relu),Dense(n,1,σ),x->3*x)
        function loss(a,b)
            Flux.Losses.mse(model(a),b)
        end
        opt = ADAM(η);
        data = Iterators.repeated((x11,y11),iter)
        ps = Flux.params(model)

        function cb!()
            push!(err_train,loss(x11,y11))
            push!(err_test, loss(xt1,yt1))
            println("loss of the function is :" ,loss(x11,y11))
        end

        Flux.train!(loss,Flux.params(model),data,opt,cb = cb!)
        pred_vec = vec(model([htd';vtd']))
        push!(all_data,pred_vec)
        # saving all the trained models having same architecture, Learningrate, etc, but different
        # training set of data
        push!(model_vec,model)
    end
end

#end # We get the data of x1,y1,xt,yt for each step and number
##
"""
Since we are training 9 different models for each set
    of data given, we will be getting different parameters for the trained model

    The error plots observed are each error corresponds to different model, trained,
    we also need to check, out of these which model gives better accuracy. while
    interpolating within the perturbed zone.
    """;

    er_train1 = reshape(err_train,iter,length(x1))
    er_test1 = reshape(err_test,iter,length(x1))

    p1 = plot()
    for i in 1:length(x1)
        p1 = plot!(er_train1[:,i],xlabel = :"Iterations",ylabel = "log10(error)",yaxis = :log10,
        label = "$(idx[i][1]) steps, $(idx[i][2])% variation ",title = "η = $(η), with $(n) Hidden units
        Iterations = $(iter)",legend= :best)
    end
    display(p1)
    f1 = string(pwd())*"\\Learning Rate $(η) Iterations = $(iter) $(n) Hidden units"
    png(f1)
    for j in 1:3
        p3 = plot()
        for i in j:3:9
            p3 = plot!(er_train1[:,i],xlabel = :"Iterations",ylabel = "log10(error)",yaxis = :log10,
            label = "$(idx[i][1]) steps, $(idx[i][2])% variation ",title = " $(idx[j][2])% variation η = $(η), with $(n) Hidden units
            Iterations = $(iter)",legend= :best)
        end
        display(p3)
        f2 = string(pwd())*"\\ $(idx[j][2])% variation Learning Rate $(η) Iterations = $(iter) $(n) Hidden units"
        png(f2)
    end

    ##
    """ Saving the model""";
    using BSON: @save, @load
    @save "model$(iter) $(η) $(n).bson" model
    plot()
    scatter(all_data)
    plot!(utd)
    ##
    """saving multiple models
    Predicting how each model perform on random traininng set""";
    @save "model_vec$(iter) $(η) $(n).bson" model_vec

    pred_vec = hcat([vec(model_vec[j]([htd';vtd'])) for j in 1:length(x1)]...)
    function L2Norm(a,b)
        Flux.mse(a,b)
    end
    l2norm_vec = [L2Norm(pred_vec[:,i],utd) for i in 1:length(x1)]

    function L∞Norm(a,b)
        maximum(abs.(a-b))
    end
    l∞norm_vec = [L∞Norm(pred_vec[:,i],utd) for i in 1:length(x1)]


    scatter(pred_vec.-utd)
    display(pred_vec[20:40,:])
    display(abs.(pred_vec[20:40,:] .-utd[20:40]))

    scatter(pred_vec)
    ##
    """ The error plots that we make are the train-test split accuracy of the model. """;



    ##
    """ Adding perturtion in between the trajectory """;
    uncertanity = [(rand()>0.5) ? 1 : 0 for _ in 1:100]

    function AnalyticalSolRand2(t,v0,h0;uncertanity = uncertanity)
        for i in 1:length(uncertanity)
            tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
            ŝ = tf̂/2 + v0/3
            if uncertanity[i] == 0
                v = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
                h = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
                u = (t<=ŝ) ? 0 : 3
                #print("rand is  < 0.5")
                return v,h,u
            elseif uncertanity[i] == 1
                v = (t<=ŝ) ? (-1.5*t + v0)*1.01 : (1.5*t -3*ŝ+v0)*1.01
                h = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0)*1.01 : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)*1.01
                u = (t<=ŝ) ? 0 : 3
                #print("rand is  > 0.5")
                return v,h,u
            end
        end

    end

    """ RK4 Integration method with perturabated function """;
    function RK4integralwithPerturbed(tf,v0,h0,n)
        global t1 = LinRange(0,tf,Int64(n))
        dt = (t1[2]-t1[1]) #we are chosing dt = 2*dt because we have u values only for selected dt range
        dh = dt;
        global uncertanity = [(rand() >0.5) ? 1 : 0 for _ in 1:length(t1)];
        # for the added uncertanity we have the analytical solution at each timestep
        v_data = [AnalyticalSolRand2(a,v0,h0)[1] for a in t1]
        h_data = [AnalyticalSolRand2(a,v0,h0)[2] for a in t1]
        u_data = [AnalyticalSolRand2(a,v0,h0)[3] for a in t1]
        tmid = [(t1[i] + t1[i+1])/2 for i in 1:length(t1)-1]

        # calculating the values at mid points for RK integration method

        h_md = [AnalyticalSolRand2(a,v0,h0)[2] for a in tmid]
        v_md = [AnalyticalSolRand2(a,v0,h0)[1] for a in tmid]
        u_md = model([h_md v_md]')
        u_pred = model([h_data v_data]')

        # This gives us data for only 100 points that we selected but for the values RK4 integration we
        # need data points at u(i+1/2)
        # h[i+1] = h[i] + (1/6)*(dh*v[i] + 4*dh*v[i+1/2] + dh*v[i+1])

        hnew = [h_data[1]]
        vnew = [v_data[1]]
        vmidnew = [v_md[1]]
        for i in 1:length(t1)-1
            vn = vnew[i] + (1/6) * ( dh*( -1.5 + u_pred[i] ) + 4* dh *( -1.5 + u_md[i]) + dh*(-1.5 + u_pred[i+1]) )
            vnew = push!(vnew , vn)
            if i < length(t1)-1
                vmd = vmidnew[i] + (1/6) * ( dh*( -1.5 + u_md[i] ) + 4* dh *( -1.5 + u_pred[i+1]) + dh*(-1.5 + u_md[i+1]) )
                vmidnew = push!(vmidnew,vmd)
            end
            hn = hnew[i] + (1/6) * ( dh * vnew[i] + 4 * dh * vmidnew[i] + dh * vnew[i+1] )
            hnew = push!(hnew , hn)
        end
        return hnew,vnew,vec(u_pred),h_data,v_data,u_data
    end


    ##
    """ We have seen that our Moon Lander has successfully trained and Verified using the RK4 method
    Now we check for accuracy with RK4 method as well as with the AnalyticalSol, change few initial
        conditions within the range and  as well as the points out of training range.
        """;
        h1 = 10.3;
        v1 = -2.03
        step = 10000
        tf1 = tfinal_(h1,v1)
        h_rk1,v_rk1,u_pred1,h_data1,v_data1,u_data1  = RK4integralwithPerturbed(tf1,v1,h1,step)

        """ Actual solution vs DNN driven trajectory with Rk4 for integration """;
        p4 = plot(t1,h_rk1,label = "DNN driven RK4 integral",ylabel = "h",
        xlabel = "time",title = "Height comparision of Actual vs DNN driven RK4 integral")
        plot!(t1,h_data1,label="actual")

        p5 = plot(t1,v_rk1,label = "DNN driven RK4 integral",
        legend=:left,ylabel = "v",xlabel = "time ",title = "velocity comparision of Actual vs DNN driven RK4 integral")
        plot!(t1,v_data1,label = "actual")

        p6 = plot(t1,u_pred1,label = "predict -> NN",legend = :topleft,
        ylabel = "u",xlabel = "time",title = "Thrust comparision of Actual vs Predicted NN")
        plot!(t1,u_data1,label = "actual")

        err = [(h_rk1[end] - h_data1[end]) (v_rk1[end] -v_data1[end]) (u_pred[end] - u_data1[end])]
        print(err)
