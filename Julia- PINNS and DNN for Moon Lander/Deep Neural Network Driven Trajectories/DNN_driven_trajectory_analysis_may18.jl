cd(@__DIR__)
using BSON: @load, @save
using Flux,Plots, Random
""" Analytical solution details""";
v0= -2.; h0 = 10.
include("MoonLanderFunctions.jl")
include("AutoML2.jl")
v0= -2.; h0 = 10.
# include("MoonLanderFunctions.jl")
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
##
""" Here we are recreating the data sets""";
for step1 in num
        for a in n1
                #0.01*(-2:0.5:2)
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
        end
end
##
""" Here we are loading the trained models for the corresponding datasets in workstation""";
@load ("model_vec.bson") model_vec
@load ("best_model_vec.bson") best_mv

##
"""Hyperparameter set obtained from the Plots titles """;
hs = [0.043099 4;0.00753 8;0.018045 4;
        0.020054 32; 0.009317 32;
        0.02014 8; 0.000613 2;
        0.007385 32; 0.001068 4;
        0.010135 32]
##
"""
Lets test for some random sample
"""
##
""" When perturbation not included in between the trajectory """;
v0 = -1.97; h0 = 10.06;
tf2 = tfinal_(h0,v0)
tf12,s11,vp1,hp1,up1 = AnalyticalSol(tf2,v0,h0)
model= best_mv[2]
""" Number of TIme STEPS""";
no = 50; # number of time steps

##
hnew,vnew,h,v = EulerInt2(tf2,no)
p1 = plot(t1,hnew,label = "Euler integral",ylabel = "h",xlabel = "time",
        title = "Height comparision of Actual vs RK4 integral")
plot!(t1,h,label="actual")
# f11 = string(pwd()*"\\DNN driven Traj Height no perturbation")
# png(f11)
p2 = plot(t1,vnew,label = "Euler integral",legend=:left,ylabel = "v",xlabel = "time ",title = "velocity comparision of Actual vs RK4 integral")
plot!(t1,v_data,label = "actual")
##
hnew,vnew,h,v = EulerIntPert2(tf2,no)
p1 = plot(t1,hnew,label = "Euler integral",ylabel = "h",xlabel = "time",
                title = "Height comparision of Actual vs RK4 integral")
plot!(t1,h,label="actual")
# f11 = string(pwd()*"\\DNN driven Traj Height no perturbation")
# png(f11)
p2 = plot(t1,vnew,label = "Euler integral",legend=:left,ylabel = "v",xlabel = "time ",title = "velocity comparision of Actual vs RK4 integral")
plot!(t1,v_data,label = "actual")
##
function RK4integral2(tf,v0,h0,n)
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
    hmidnew = [h_md[1]]
    up =[];
    global  hnew,vnew,u_pred,u_md
    for i in 1:length(t1)-1
        # u_pred = reduce(vcat,model([last(hnew),last(vnew)]))
        u_pred = reduce(hcat,model([last(hnew),last(vnew)]))
        u_md = reduce(hcat,model([last(hmidnew),last(vmidnew)]))
        up = push!(up,u_pred)
        # print(vnew,hnew,u_pred)
        vn = vnew[i] + (1/6) * ( dh*( -1.5 + u_pred ) + 4* dh *( -1.5 + u_md) + dh*(-1.5 + u_pred) )
        vnew = push!(vnew ,vn)
        if i < length(t1)-1
            u_md = reduce(vcat,model([last(hmidnew),last(vmidnew)]))
            vmd = vmidnew[i] + (1/6) * ( dh*( -1.5 + u_md ) + 4* dh *( -1.5 + u_pred) + dh*(-1.5 + u_md) )
            vmidnew = push!(vmidnew,vmd)
        end
        hn = hnew[i] + (1/6) * ( dh * vnew[i] + 4 * dh * vmidnew[i] + dh * vnew[i+1] )
        hnew = push!(hnew , hn)
    end
    return hnew,vnew,up,h_data,v_data,u_data
end

h_rk,v_rk,u_pred,h_data,v_data,u_data  = RK4integral2(tf2,v0,h0,no)
display([h_rk h_data])
p1 = plot(t1,h_rk,label = "RK4 integral",ylabel = "h",xlabel = "time",title = "Height comparision of Actual vs RK4 integral")
plot!(t1,h_data,label="actual")
# f11 = string(pwd()*"\\DNN driven Traj Height no perturbation")
# png(f11)
p2 = plot(t1,v_rk,label = "RK4 integral",legend=:left,ylabel = "v",xlabel = "time ",title = "velocity comparision of Actual vs RK4 integral")
plot!(t1,v_data,label = "actual")
# f11 =
png(string(pwd()*"\\DNN driven Traj velocity no perturbation"))
# p3 = plot(t1,u_pred,label = "predict -> NN",legend = :topleft,ylabel = "u",xlabel = "time",title = "Thrust comparision of Actual vs RK4 integral")
plot!(t1,u_data,label = "actual")
png(string(pwd()*"\\DNN driven Traj Thrust no perturbation"))

# """ WHen perturbation included in trajectory""";
# v1= v0;h1 = h0;
# tf1 = tfinal_(h1,v1)
# step1 = 1000
# h_rk1,v_rk1,u_pred1,h_data1,v_data1,u_data1  = RK4integralwithPerturbed(tf1,v1,h1,step1)
# display([h_rk1 h_data1])
#
# """ Actual solution vs DNN driven trajectory with Rk4 for integration """;
# p4 = plot(t1,h_rk1,label = "DNN driven RK4 integral",ylabel = "h",
#     xlabel = "time",title = "Height comparision of Actual vs DNN driven RK4 integral")
# plot!(t1,h_data1,label="actual")
#
# p5 = plot(t1,v_rk1,label = "DNN driven RK4 integral",
#     legend=:left,ylabel = "v",xlabel = "time ",title = "velocity comparision of Actual vs DNN driven RK4 integral")
# plot!(t1,v_data1,label = "actual")
#
# p6 = plot(t1,u_pred1,label = "predict -> NN",legend = :topleft,
#     ylabel = "u",xlabel = "time",title = "Thrust comparision of Actual vs Predicted NN")
# plot!(t1,u_data1,label = "actual")
#
# err = [(h_rk1[end] - h_data1[end]) (v_rk1[end] -v_data1[end]) (u_pred[end] - u_data1[end])]
# print(err)
# ##
# display([h_rk h_rk1])
