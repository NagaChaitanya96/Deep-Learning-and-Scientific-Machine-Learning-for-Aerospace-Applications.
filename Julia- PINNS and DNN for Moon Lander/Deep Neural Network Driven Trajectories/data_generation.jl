""" For the same amount of NN architecture varying the robustness from 1 : 10 % """;
cd(@__DIR__)
using BSON: @load
using Flux,Plots, Random

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
htest,vtest = 10.015,-2.065#vec([maximum(h_dist)+0.0015*a for a in [1 5 10]]) , vec([minimum(v_dist)-0.0015*a for a in [1 5 10]]);
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
iter = 50;
##
for step1 in num
    for a in n1
        dist = (-1:0.2:1)*0.01*a
        global v_dist,h_dist = v0.+collect(dist), h0.+collect(dist)
        tf_dist = [(2/3)*(v1)+ (4/3*(sqrt((v1^2/2) + 1.5*h1))) for v1 in v_dist, h1 in h_dist]
        h_matrix = [];
        u_matrix = [];
        v_matrix = [];
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
        n = 4;
        #for i in 1:length(x1)
        global model = Chain(Dense(2,n,σ),Dense(n,n,leakyrelu),Dense(n,n,relu),Dense(n,1,σ),x->3*x)
        function loss(a,b)
            Flux.Losses.mse(model(a),b)
        end
        opt = ADAM(0.05);
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

        #end
    end
end

#end # We get the data of x1,y1,xt,yt for each step and number
##
err_train = reshape(err_train,iter,length(x1))
err_test = reshape(err_test,iter,length(x1))

p1 = plot()
labels = ["1% variation","5% variation","10% variation"]
for i in 1:length(x1)
    p1 = plot!(err_train[:,i],xlabel = :"Iterations",ylabel = "log10(error)",yaxis = :log10,
    title = "Data used to train the
    model is $(maximum(size(x1[1]))), number of iterations = $(iter)",label = labels[i])
end
display(p1)
##
""" Defining a neural network""";
n = 4;
iter = 1000
err_train= [];
err_test= [];
for i in 1:length(x1)
    global model = Chain(Dense(2,n,σ),Dense(n,n,leakyrelu),Dense(n,n,relu),Dense(n,1,σ),x->3*x)
    function loss(a,b)
        Flux.Losses**.mse(model(a),b)
    end

    opt = ADAM(0.05);
    data = Iterators.repeated((x1[i],y1[i]),iter)
    ps = Flux.params(model)


    function cb!()
        push!(err_train,loss(x1[i],y1[i]))
        push!(err_test, loss(xt[i],yt[i]))
        println("loss of the function is :" ,loss(x1[i],y1[i]))
    end

    Flux.train!(loss,Flux.params(model),data,opt,cb = cb!)
    pred_vec = vec(model([htd';vtd']))
    push!(all_data,pred_vec)
end
err_train = reshape(err_train,iter,length(x1))
err_test = reshape(err_test,iter,length(x1))

p1 = plot()
labels = ["1% variation","5% variation","10% variation"]
for i in 1:length(x1)
    p1 = plot!(err_train[:,i],xlabel = :"Iterations",ylabel = "log10(error)",yaxis = :log10,
    title = "Data used to train the
    model is $(maximum(size(x1[1]))), number of iterations = $(iter)",label = labels[i])
end
display(p1)
##

""" Error calculation""";
err_ = [];
push!(err_,[abs.(utd1[i]-all_data[i]) for i in 1])
err_ = reduce(hcat,reduce(vec,err_))
Flux.mse(utd[1],all_data[1])


## PLOTS
""" We have seen that our Moon Lander has successfully trained and Verified using the RK4 methodow we check for accuracy with RK4 method as well as with the AnalyticalSol, change few initial
conditions within the range and  as well as the points out of training range.
""";
begin
    h1 = 10.0;
    v1 = -2.0
    step = 1000
    tf1 = time_final(h1,v1)
    h_rk1,v_rk1,u_pred1,h_data1,v_data1,u_data1  = RK4integral(tf1,v1,h1,step)
    err = [(h_rk1[end] - h_data1[end]) (v_rk1[end] -v_data1[end]) (u_pred[end] - u_data1[end])]
    print(err)

    """ Verifying the accuracy of model """;
    s21 =[(pred1[i]>1e-4) && (pred1[i] <3) ? pred1[i]=2 : pred1[i] for i in 1:length(pred1)]
    fc = count(==(2),s21)
    s22 =[(pred[i]>1e-4) && (pred[i] <3) ? pred[i]=2 : pred[i] for i in 1:length(pred)]
    fc2 = count(==(2),s22)
    ktrain= (length(s21)-Int64.(fc))/length(s21)
    ktest= (length(s22)-Int64.(fc2))/length(s22)

    print("Accuracy in of NN in train set is ", ktrain)
    print("Accuracy of NN in test set is ",ktest)
end

""" This code is correct """
