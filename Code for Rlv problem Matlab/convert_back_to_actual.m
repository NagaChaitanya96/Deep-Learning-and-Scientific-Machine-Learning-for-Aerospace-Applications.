function [r_act,v_act,s_act,h_act,m_act,time_act,gamma_act,...
    theta_act,omega_act,Thrust_act,beta_act] = convert_back_to_actual()
global R0 g0 m0 s0 v0 h0 omega0 states solg Cd S_ref Isp I_z l_com t_ref st break_time guess_matrix 
global gamma0 omega0 theta0 controls
t_reference = 32;
r1 = states.r;
v1 = states.V;
gamma1 = states.gamma;
m1 = states.m;
omega1 = states.omega;
s1 = states.s;
theta1 = states.theta;

time1 = controls.time_interp;
T1 = controls.TT_interp;
beta1 = controls.beta_interp;

%% Converting back to actual forms
r_act = r1.*R0;
v_act = v1.*v0;
s_act = s1.*s0;
h_act = r_act- R0;
m_act = m1.*m0;
time_act = time1.*t_reference;
gamma_act = gamma1;
theta_act = theta1;
omega_act = omega1;

Thrust_act = T1.*(m0*g0);
beta_act = beta1;
end 
