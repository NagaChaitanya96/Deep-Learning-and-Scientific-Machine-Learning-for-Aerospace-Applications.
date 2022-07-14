# "D:\\2022\\March\\RLV data generation\\Data Training\\Moon Lander\\MARCH 19\\21th data"
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
""" We have saved the models in an order, so we will be sending the data for each particular model
and finding the loss function of that particular model and save in a vector to plot the variation
between different perturbation %s """;

loss_vec = [];
""" For 10 randomly sampled learning rates and Number of Hidden units we have observed 90 models """;
for j in 1:10
        for i in 1:9
                model = model_vec[i*j]
                pred1 = model(x1[i])
                # loss_mse(pred1,y1[1])
                pred2 = vec(pred1)
                yact = vec(y1[i])
                err_v = Flux.mse(pred2,yact)
                push!(loss_vec,err_v)
        end
end
#= now our loss vec is the loss value of each model cooresponding to each data set. Now we reshape it
such that for 10 randomly choosen n, η, we will obtain losses correspondingly.
=#
""" This loss matrix has values for corresponding (idx) for 10 different n, η """
loss_matrix = Float64.(reshape(loss_vec,9,10))
# loss_matr2 = loss_matrix'
""" Data for 1,5,10% variation for seperate n,η as matrix """;
onepercent_loss_with_var_in_points = reshape(reduce(vcat, [[loss_matrix[:,j][i] for i in 1:3:9] for j in 1:10]),3,10)
Fivepercent_loss_with_var_in_points =reshape(reduce(vcat, [[loss_matrix[:,j][i] for i in 2:3:9] for j in 1:10]),3,10)
TENpercent_loss_with_var_in_points =reshape(reduce(vcat, [[loss_matrix[:,j][i] for i in 3:3:9] for j in 1:10]),3,10)

""" PLOTS FOR EACH VARIATION """;
# s2 = collect(-8:-1)
# ytick_vec = [10.0^i for i in s2]
list = ["1% variation", "5% variation", "10% variation"]
for i in 1:10
        pl = plot()
        plot!(xlabel = "Number of points selected in trajectory 100,500,1000")
        plot!(yaxis = ("Training Loss(MSE)", (1e-7,1e-1),:log10))
        plot!(title = "Hyperparameters n = $(hs[i,2]) and η = $(hs[i])
        iterations = 7500")
        scatter!(onepercent_loss_with_var_in_points[:,i],label =list[1])
        scatter!(Fivepercent_loss_with_var_in_points[:,i],label = list[2])
        scatter!(TENpercent_loss_with_var_in_points[:,i],label = list[3])
        plot!(legend=:bottomright)
        display(pl)
        f4 =string(pwd()*"\\Hyperparameters n = $(hs[i,2]) and η = $(hs[i]) iterations = 7500")
        png(f4)
end

# plot(;xticks = log10.([100,500,1000]),xaxis =:log10)
# scatter([onepercent_loss_with_var_in_points[:,i] for i in 1:2],
#         xlims = (100,500),xticks = 100:400:1000,
#         yticks = ytick_vec,yaxis = :log10,ylabel = "Training Loss (MSE)",
#         legend= :bottom)

# scatter([onepercent_loss_with_var_in_points[:,i] for i in 1:2], xaxis = ("my label", (0,3), 0:1:10, :log))

##
""" Converting to data frame for better representation """
using DataFrames
# x = rand(2,2)
# convert(DataFrame,x)
# convert(DataFrame,loss_matrix)
# data1 = DataFrame(CSV.read("RLV_data_best.csv",DataFrame,header = ["H","V","S","ω","γ","m","θ","Thrust","β"]))
""" After converting to data frame, plot the variation of n,η for different perturbation and number of points """;
##
""" PLOTS""";
# plot(sort(vec(m1(x1[1]))))
# plot!(sort(vec(y1[1])))
##
""" TESTING DATA DRIVEN TRAJECTORY OF THE MODEL. """;
