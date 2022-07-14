"""   Finguring out best possible NN ACtivation functions for MOON LANDER PROb.
""";
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

""" Analytical solution details""";
v0= -2.; h0 = 10.
tfinal1(h0,v0) = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
tf = tfinal1(h0,v0)
function AnalyticalSol(t,v0,h0)
     tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
     ŝ = tf̂/2 + v0/3
     v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
     ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
     û(t) = (t<=ŝ) ? 0 : 3
     return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
end

##
""" Now vary the details by 5% in v0 and h0 in 11 steps
V,H disturbance = [v0,h0] ±5% """;

dist = (-5:1:5)*0.01
v_dist,h_dist = v0.+collect(dist), h0.+collect(dist)
# v_dist will be available in the rows and h_dist will be available in columns
tf_dist = [(2/3)*(v1)+ (4/3*(sqrt((v1^2/2) + 1.5*h1))) for v1 in v_dist, h1 in h_dist]

##
h_matrix = [];
u_matrix = [];
v_matrix = [];
""" For different values of h we get matrix of h,u,v w.r.t time and velocity
h = 9.95 -> h0 in h_dist[1]
v = -2.05 -> v0 in v_dist[1]

1. Now perturbing from 11*11 in H and V we will be obtaining different values, without converting
     to vector we will have 40*11*11 matrix.

2. Now our data has repeated points and corresponding values.

3. We will sort our data into ascending order and seperate out the repeated data, to make sure our Neural Network
     parameters will be consistent without any excess weightage given to the repeated points.

4. Now for each data point variation we are calculating the results for moon lander for 100 time steps
""";

h_mat = [];
v_mat = [];
u_mat = [];
t_all = [];

for v0 in v_dist
     for h0 in h_dist
                 tfinal = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
                 h_data = [AnalyticalSol(a,v0,h0)[4] for a in LinRange(0,tfinal,10)]
                 v_data = [AnalyticalSol(a,v0,h0)[3] for a in LinRange(0,tfinal,10)]
                 u_data = [AnalyticalSol(a,v0,h0)[5] for a in LinRange(0,tfinal,10)]

                 h_matrix = push!(h_matrix,h_data)
                 v_matrix = push!(v_matrix,v_data)
                 u_matrix = push!(u_matrix,u_data)

                 h_mat = hcat(h_matrix...)
                 v_mat = hcat(v_matrix...)
                 u_mat = hcat(u_matrix...)
                 t_all = push!(t_all,tfinal)
     end
end

""" To check we have the data for all the v0 and h0 at all times we choose to verify the time """;
t_all == vec(tf_dist') # this states that for each combination of v0,h0 we have enough data and that is true.

##
""" Now that we have the necessary data we choose the package to get the approximate neural network trained """;
using Flux
#=
Now what is the input and output?

We need to train a NN such that our params will be able to predict minimum fuel consumption at given states, since our
objective function is minimizing the U.

1. Define a neural network, with Hidden Units and Layers
2. divide the set of data to train and test sets.     (randomize the dataset)
3. define a cost function which should be minimized
4. train the weights and params for the cost function.
5. select the cross validation set to check the accuracy of model if, its high bias model or high variance model.
6. If the model is high bias, we need to add more features. For high variance we need to add more training data.

=#

## SORTING OUR DATA H0 AND V0 AS INPUT AND U AS OUTPUT
h_data  = vec(h_mat)
v_data  = vec(v_mat)
u_data  = vec(u_mat)
ip = [h_data' ; v_data']
op = hcat(u_data...)
""" DONT USE SHUFFLE , BECAUSE IT SHUFFLES ALL THE DATA IN ENTIRE MATRIX.""";
using Random
ss = Int64.(0.70*length(u_data))
rng = MersenneTwister(1234)
""" Shuffling done here""";
numbers = shuffle(rng,Vector(1:length(u_data)))
data2 = [h_data v_data u_data]
data2 = [data2[i,:] for i in eachrow(numbers)]
data2 = reduce(vcat,data2)
##
""" Setting to  test and train sets """;
X = data2[:,[1,2]]'
Y = data2[:,3]'
x1 , y1 = X[:,1:ss] , Y[:,1:ss]
xt, yt = X[:,ss+1:end] , Y[:,ss+1: end]
##
n = 16;
#model = Chain(Dense(2,n,σ),Dense(n,n,leakyrelu),Dense(n,n,leakyrelu),Dense(n,1,relu))
#model = Chain(Dense(2,n,σ),Dense(n,n,leakyrelu),Dense(n,n,leakyrelu),Dense(n,1,σ),x->3*x) # This model has 82% accuracy in train and test set
model = Chain(Dense(2,n,σ),Dense(n,n,leakyrelu),Dense(n,n,relu),Dense(n,1,σ),x->3*x)
#model = Chain(Dense(2,n,leakyrelu),Dense(n,n,leakyrelu),Dense(n,n,leakyrelu),Dense(n,1,relu))
function loss(a,b)
     Flux.Losses.mse(model(a),b)
end


opt = ADAM(0.0005);
data = Iterators.repeated((x1,y1),5000)
ps = Flux.params(model)

err_train= [];
err_test= [];
function cb!()
     push!(err_train,loss(x1,y1))
     push!(err_test, loss(xt,yt))
     println("loss of the function is :" ,loss(x1,y1))
end
@time Flux.train!(loss,Flux.params(model),data,opt,cb =cb!)

using  BSON: @save , @load
@save("model10.bson",model)




##
pred1 = vec(model(x1))
pred = vec(model(xt))
s21 =[(pred1[i]>1e-4) && (pred1[i] <2.99) ? pred1[i]=2 : pred1[i] for i in 1:length(pred1)]
fc = count(==(2),s21)
s22 =[(pred[i]>1e-4) && (pred[i] <2.99) ? pred[i]=2 : pred[i] for i in 1:length(pred)]
fc2 = count(==(2),s22)
ktrain= (length(s21)-Int64.(fc))/length(s21)
ktest= (length(s22)-Int64.(fc2))/length(s22)

print("Accuracy in of NN in train set is ", ktrain)
print("Accuracy of NN in test set is ",ktest)
## Test set common for all the variation in number of points
function AnalyticalSol(t,v0,h0)
     tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
     ŝ = tf̂/2 + v0/3
     v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
     ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
     û(t) = (t<=ŝ) ? 0 : 3
     return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
end

h_testf = 10.06; v_testf = -2.06;
tf_test = tfinal1(h_testf,v_testf)
htest,vtest,utest = AnalyticalSol(tf_test,v_testf,h_testf)



## PLOTS
scatter(pred,label = "predicted")
scatter!(vec(yt),label = "actual")

plot(log10.(err_train))
plot!(log10.(err_test))
ps = Flux.params(model)

##
hnew=[];
vnew=[];
unew=[];
""" Now we are going to the evaluate the how the values of h,v are obtained for the values of u that
we send into the Euler integration. So for the analytical solution of h0,v0 ,we have a trained model
which will send the values of u, for our initial conditions.

Now using the u values as our initial guess and at each time step we are running the Euler SOlver
forward in time so that, we will estimate the desired h and v .

""";

""" Now to make sure we have the right accuracy we test our model using the EUler integration strategy""";
v0= -1.95; h0 = 10.5
tfinal  = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))

h = [AnalyticalSol(a,v0,h0)[4] for a in LinRange(0,tfinal,10)]
v = [AnalyticalSol(a,v0,h0)[3] for a in LinRange(0,tfinal,10)]

u = model([h';v'])
hnew =[];
vnew = [];

##

""" Euler Integration method """;
no  = 1e6
function EulerInt(tfinal,n)
      time1 = 0:1/n:tfinal
      h = [AnalyticalSol(a,v0,h0)[4] for a in time1]
      v = [AnalyticalSol(a,v0,h0)[3] for a in time1]
      u = model([h';v'])
      dt = time1[2]-time1[1]
      hnew = [h0]; vnew = [v0];
      for i in 1:length(time1)-1
            hn = hnew[i] + dt * vnew[i]
            vn = vnew[i] + dt * (-1.5+u[i])
            hnew = push!(hnew,hn)
            vnew = push!(vnew, vn)

      end
      return hnew,vnew,h,v
end

hnew,vnew,h,v= EulerInt(tf,no)

""" RK4 Integration method""";
function RK4integral(tf,v0,h0,n)
    global t1 = LinRange(0,tf,Int64(n))
    dt = (t1[2]-t1[1]) #we are chosing dt = 2*dt because we have u values only for selected dt range
    dh = dt;
    h_data = [AnalyticalSol(a,v0,h0)[4] for a in t1]
    v_data = [AnalyticalSol(a,v0,h0)[3] for a in t1]
    u_data = [AnalyticalSol(a,v0,h0)[5] for a in t1]

    tmid = [(t1[i] + t1[i+1])/2 for i in 1:length(t1)-1]

    # calculating the values at mid points for RK integration method

    h_md = [AnalyticalSol(a,v0,h0)[4] for a in tmid]
    v_md = [AnalyticalSol(a,v0,h0)[3] for a in tmid]
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

h_rk,v_rk,u_pred,h_data,v_data,u_data  = RK4integral(tf,v0,h0,no)

p1 = plot(t1,h_rk,label = "RK4 integral",ylabel = "h",xlabel = "time",title = "Height comparision of Actual vs RK4 integral")
plot!(t1,h_data,label="actual")

p2 = plot(t1,v_rk,label = "RK4 integral",legend=:left,ylabel = "v",xlabel = "time ",title = "velocity comparision of Actual vs RK4 integral")
plot!(t1,v_data,label = "actual")

p3 = plot(t1,u_pred,label = "predict -> NN",legend = :topleft,ylabel = "u",xlabel = "time",title = "Thrust comparision of Actual vs RK4 integral")
plot!(t1,u_data,label = "actual")

## PLOTS
""" We have seen that our Moon Lander has successfully trained and Verified using the RK4 method
      Now we check for accuracy with RK4 method as well as with the AnalyticalSol, change few initial
      conditions within the range and  as well as the points out of training range.
""";
h1 = 10.0;
v1 = -2.0
step = 10000
tf1 = time_final(h1,v1)
h_rk1,v_rk1,u_pred1,h_data1,v_data1,u_data1  = RK4integral(tf1,v1,h1,step)


""" Actual solution vs DNN driven trajectory with Rk4 for integration """;
p1 = plot(t1,h_rk1,label = "DNN driven RK4 integral",ylabel = "h",xlabel = "time",title = "Height comparision of Actual vs DNN driven RK4 integral")
plot!(t1,h_data1,label="actual")

p2 = plot(t1,v_rk1,label = "DNN driven RK4 integral",legend=:left,ylabel = "v",xlabel = "time ",title = "velocity comparision of Actual vs DNN driven RK4 integral")
plot!(t1,v_data1,label = "actual")

p3 = plot(t1,u_pred1,label = "predict -> NN",legend = :topleft,ylabel = "u",xlabel = "time",title = "Thrust comparision of Actual vs Predicted NN")
plot!(t1,u_data1,label = "actual")

err = [(h_rk1[end] - h_data1[end]) (v_rk1[end] -v_data1[end]) (u_pred[end] - u_data1[end])]
print(err)
