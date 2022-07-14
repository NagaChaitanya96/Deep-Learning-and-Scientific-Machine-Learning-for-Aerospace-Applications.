clc;
close all;
clear all;
format longG;
%%
global R0 g0 m0 s01 v01 h01   omega0 states solg Cd S_ref Isp I_z l_com t_ref st break_time guess_matrix time_guess
global gamma0 omega0 theta0 controls point_each
R0=6378*10^3; % Earth's radius
g0=9.81;
S_ref=10.75;
Cd=0.5;
Isp=300;
I_z=3346393;
l_com=9.778; % centre of mass distance
t_ref=32;

h01=5000;        % No variation accepted %5000
s01=850;         % 850 and  855 are only working for this downrange distance %850
v01=325;         % acceptable range 325 -351 %325
gamma0 = -75*pi/180; %initial 75 changing gamma0 and theta0 in equal ranges gives good results
% whereas compared to individual variation in both the initial angles.
m0=26229.667;
theta0 = -75*pi/180; % initial 75
omega0 = 0.342;
%% Progress in Data generated
% 1. Generated data for 1,2,5,10% variation each having 100 points and 729
% trajectories. approximately we have 303 points per trajectory as we have
% two segments for each trajectory, which gives us 221291 state action
% % pairs.
% 1.2 Try 303 trajectories with 730 points in between for  this stepsize should
% be considered different. 
%% 15TH MARCH
%  BUT TO ATTAIN THIS WE NEED 6 SEGMENTS AND IN BETWEEN EACH
% SEGEMENT CONTAINTS 101 POINTS. HOWEVER, BREAKING THE ALGORITHM TO MANY
% SEGMENTS IS NOT FEASIBLE FOR OPTIMIZATION PURPOSES, SINCE OUR OBJECTIVE
% WAS TO ACHIEVE TRAJECTORY WITH AS LESS NUMBER OF SEGMENTS AND POINTS IN
% BETWEEN. THEREFOR INSTEAD OF TRYING THIS METHOD, WE WOULD LIKE TOR TRY
% VARYING THE NUMBER OF STEP TO [50, 500, 1000].
%%
% 2. should try for 3300 traj and 303 s-a points per trajectory to get a million s-a pairs
% 3. should try for 330 traj and 3003 s-a pairs per trajectory to get a million s-a pairs.
% 
% 4. normalize the data and train NN with different Hyper params.
% 5. check which one does better




%% BEFORE MAKING ANY CHANGE IN ANY OTHER VARIABLE MAKE SURE U CREATE A INITIAL GUESS AS M01,H01 ETC...
range = 0.1*(-0.5:0.2:0.5); %9 steps within 10 percent variation is collected
% now need to collect data for 1,5,10% with same steps
% additional steps implemented are taken 2 break points  within the time
% Now we will check what happens if we increase the number of steps
% f11 = [0.0025151, 0.012822222255555  ,0.02631555555 -> 1,5,10 percent variation
f11 = [0.0025151, 0.012822222255555  ,0.02631555555]
range =  0.02631555555*(-2:0.5:2);
hvec = h01 + h01*range;
vvec = v01 + v01*range;
svec = s01 + s01*range;
hdata = [];
vdata =[];
sdata = [];
rdata = [];
omegadata = [];
gammadata =[];
mdata = [];
thetadata = [];
Thrustdata = [];
betadata = [];
timedata= [];
hvar = (max(hvec)-min(hvec))/max(hvec)*100
svar = (max(svec)-min(svec))/max(svec)*100
vvar = (max(vvec)-min(vvec))/max(vvec)*100

L = 1;
ic = [];
data_vec = [];
for i = 1:length(hvec)
    h0 = hvec(i);
    for j = 1:length(vvec)
        v0 = vvec(j);
        for k = 1:length(svec)
            s0 = svec(k);
            ic = [ic,[L h0 v0 s0]'];
            dvec = [L h0 v0 s0];
            data_vec  = [ data_vec; dvec];
            L = L+1;
        end
    end
end
max(size(data_vec))
%%
hstep = 100;
%%
sol_vec = [];
gm= load('guess_matrix1.mat');
tg = load('time_guess1.mat');
time_guess = tg.time_guess;
%hstep = 100; % we expect 303 points per traj since we have 2 segments
bt = load('breaktime1.mat');
pe = load('point_each1.mat');
%%
% index numbers that i have skipped 

%%
for l = 1:max(size(data_vec))
    time_guess = tg.time_guess;
    break_time = bt.break_time;
    point_each = pe.point_each;
    guess_matrix = gm.guess_matrix;
    
    
    h0 = data_vec(l,2);
    v0 = data_vec(l,3);
    s0 = data_vec(l,4);
    L = data_vec(l,1);
    disp([L h0 s0 v0])
    [r_act,v_act,s_act,h_act,m_act,time_act,gamma_act,...
        theta_act,omega_act,Thrust_act,beta_act] = RLV_p_k_adaptive(h0,v0,s0,hstep);
    hdata = [hdata,h_act];
    vdata = [vdata,v_act];
    sdata = [sdata,s_act];
    omegadata = [omegadata,omega_act];
    gammadata = [gammadata,gamma_act];
    mdata = [mdata,m_act];
    thetadata = [thetadata,theta_act];
    Thrustdata = [Thrustdata,Thrust_act];
    betadata = [betadata,beta_act];
    timedata = [timedata,time_act];    
    dvec = [h0;v0;s0];
    %data_vec  = [ data_vec, dvec];
    
    solc = [hdata;vdata;sdata;omegadata;gammadata;mdata;thetadata;Thrustdata;betadata;timedata];
    sol_vec = solc;
    L = L+1;
    global solF
    solF = [hdata',vdata',sdata',omegadata',gammadata',mdata',thetadata',Thrustdata',betadata',timedata'];
    csvwrite('May7rd10percent_RLV_data 100 points2 .csv',solF)
    save('May7rd10percent_RLV_data_100_points2.mat','hdata','vdata','sdata','omegadata','gammadata',...
        'mdata','thetadata','Thrustdata','betadata','timedata')
end
%% for each 1:30003 points we have the solutions of state-action pairs for 1 trajectory
% solF = [hdata',vdata',sdata',omegadata',gammadata',mdata',thetadata',Thrustdata',betadata'];
% %%
% save('RLV_data.mat','hdata','vdata','sdata','omegadata','gammadata',...
%     'mdata','thetadata','Thrustdata','betadata')
%
% %% saving in csv file
% csvwrite('RLV_data.csv',solF)
% data_now = [h_act',v_act',s_act',omega_act',gamma_act',m_act',theta_act',Thrust_act',beta_act'];
% first_row = data_now(1,: )
% 
% % solF = [hdata',vdata',sdata',omegadata',gammadata',mdata',thetadata',Thrustdata',betadata'];
% % solF(303*1,:)~=solF(303*2,:)
% % csvwrite('fr.csv',first_row)
% 
% 














