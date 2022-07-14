function [r_int,s_int,V_int,gamma_int,m_int,theta_int,omega_int]=rk4_integral(timef,point_each,r,s,V,gamma,m,theta,omega,T,beta)
global R0 g0 m0 s0 v0 h0   omega0 states solg S_ref Cd I_z l_com t_ref Isp hstep
time=timef; % timef  = time_ch == time_final
y0=[r(1) s(1) V(1) gamma(1) m(1) theta(1) omega(1)]; % entire state vectors Y'
y0=y0'; % convert the values into the vector form and this is sent to the diff_eqn2 to solve RK4 discretely.
% here the sent in values will be time_interp
N =hstep;
h=(time(end)-time(1))/N; % stepsize
t=time(1):h:time(end); % 0.0 :0.0001:0.860882898
y(:,1) = y0;                                          % initial condition

% since at time t0 we know the values we need to find the integrated steps
%   from t1 to tfinal which is in total length(time_interp) - 1

% Now since RK method is y0+h/6*(k1+2k2+2k3+k4) our K1,K2,K3,K4 values
% k1 = f(t0,y0)
%% t0 = 0; y0 obtained from states(1) and diff_eqn2(t0,y0,T,beta,time_ch)
% diff_eqn2(t0,y0,T,beta,time_ch)

for i=1:(length(t)-1)
    k_1 = diff_eqn2(t(i),y(:,i),T,beta,time); % f(t0,y0, T_ )
    k_2 = diff_eqn2(t(i)+0.5*h,y(:,i)+0.5*h*k_1,T,beta,time);
    k_3 = diff_eqn2((t(i)+0.5*h),(y(:,i)+0.5*h*k_2),T,beta,time);
    k_4 = diff_eqn2((t(i)+h),(y(:,i)+k_3*h),T,beta,time);
    y(:,i+1) = y(:,i) + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;  % main equation
end
ynew = y;   % since y keeps on changing we implement the ynew to store 
% the entire set of values
% this Y(:,1) is a column vector (7-by-1) of all the states; y(t0) = y0. so
% after N steps we will be having N*7 matrix of y with all states
% interpolated through time.

% now the values obtained here are interpolated at each time step and the
% states are found out using the time stepping method from the above y
% matrix. For this matrix we use the controls T, beta. (ARE THEY GUESSED
% VALUES).

% If we have the accurate values of controls in 1 iteration, I will be
% getting the accurate values of states through this method. SO OUR THRUST
% AND BETA CANNOT BE GUESSED VALUES. They should be the optimized values.

%% Calculating for mid point accuracy
for i=1:point_each-1
    time_mid(i)=0.5*(time(i)+time(i+1));    % (time_changed(1)+ time_changed(2)) * 0.5
end

% from the N*7 matrix in y we will have seperated values for states and are
% obtained using the r_states = y(1,:). The below interpolation is for
% midpoints 

% states.r = ynew(1,:);
% states.s = ynew(2,:);
% states.V = ynew(3,:);
% states.gamma = ynew(4,:);
% states.m = ynew(5,:);
% states.theta = ynew(6,:);
% states.omega = ynew(7,:);
% states.time = t;

states.r = [states.r, ynew(1,:)];
states.s = [states.s ,ynew(2,:)];
states.V = [states.V , ynew(3,:)];
states.gamma = [states.gamma ,ynew(4,:)];
states.m = [states.m, ynew(5,:)];
states.theta = [states.theta, ynew(6,:)];
states.omega = [states.omega,ynew(7,:)];
states.time = [states.time,t];


r_int=interp1(t,y(1,:),time_mid)';
s_int=interp1(t,y(2,:),time_mid)';
V_int=interp1(t,y(3,:),time_mid)';
gamma_int=interp1(t,y(4,:),time_mid)';
m_int=interp1(t,y(5,:),time_mid)';
theta_int=interp1(t,y(6,:),time_mid)';
omega_int=interp1(t,y(7,:),time_mid)';

%  r_int=lagrange(time_mid,t,y(1,:))';
%  s_int=lagrange(time_mid,t,y(2,:))';
%  V_int=lagrange(time_mid,t,y(3,:))';
%  gamma_int=lagrange(time_mid,t,y(4,:))';
%  m_int=lagrange(time_mid,t,y(5,:))';
%  theta_int=lagrange(time_mid,t,y(6,:))';
%  omega_int=lagrange(time_mid,t,y(7,:))';

%   r_int=loren(1,:)';
%   s_int=loren(2,:)';
%   V_int=loren(3,:)';
%   gamma_int=loren(4,:)';
%   m_int=loren(5,:)';
%   theta_int=loren(6,:)';
%   omega_int=loren(7,:)';
end