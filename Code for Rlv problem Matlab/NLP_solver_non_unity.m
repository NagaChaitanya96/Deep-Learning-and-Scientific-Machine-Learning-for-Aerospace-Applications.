function [time_break,rr,ss,VV,gammag,mm,thetat,omegao,...
    TT,betab]=NLP_solver_non_unity(point_each,break_time,guess_matrix,time_guess)
global R0 g0 m0 s0 v0 h0 theta0 gamma0 omega0 solg init prob

coll_points=sum(point_each)-(length(point_each)-1);
S_ref=10.75;
ri=(R0+h0)/R0;
%rf=1 as the rocket will have landed.

I_z=3346393;
l_com=9.778; % centre of mass distance
T_max=756.222e3/(m0*g0);

% Defining optimisation variables
r=optimvar('r',coll_points,1);
s=optimvar('s',coll_points,1);
V=optimvar('V',coll_points,1);
gamma=optimvar('gamma',coll_points,1);
m=optimvar('m',coll_points,1);
theta=optimvar('theta',coll_points,1);
omega=optimvar('omega',coll_points,1);
tau_f=optimvar('tau_f',1,1);

% Control variables
T=optimvar('T',coll_points,1);
beta=optimvar('beta',coll_points,1);


T_breaks=optimvar('T_breaks',1,length(break_time));
time_breaks_2=[0 T_breaks]; % best result with this
time_breaks=[time_breaks_2,tau_f];


init.T_breaks=break_time;
break_time_add=[0 break_time];
% disp(point_each)

% if max(size(sol_guess.time)) >= max(size(time_guessed))
%     fprintf(" size @ left is > right")
% else 
%     fprintf("size @right is < left")
% end


[r_init,s_init,V_init,gamma_init,m_init,theta_init,...
    omega_init,T_init,beta_init,t_end_g]=initial_generate_RLV(break_time_add,point_each,guess_matrix,time_guess);

if max(size(T)) ~= max(size(T_init))
    no = abs(max(size(T)) - max(size(T_init)));
    init.tau_f=t_end_g;
    init.r=r_init(1:end-no)';
    init.s=s_init(1:end-no)';
    init.V=V_init(1:end-no)';
    init.gamma=gamma_init(1:end-no)';
    init.m=m_init(1:end-no)';
    init.theta=theta_init(1:end-no)';
    init.omega=omega_init(1:end-no)';
    init.T=T_init(1:end-no)';
    init.beta=beta_init(1:end-no)';
else
    init.tau_f=t_end_g;
    init.r=r_init.';
    init.s=s_init.';
    init.V=V_init.';
    init.gamma=gamma_init.'                 ;
    init.m=m_init.';
    init.theta=theta_init.';
    init.omega=omega_init.';
    init.T=T_init.';
    init.beta=beta_init.';
end




%% Solving NLP
prob=optimproblem('Objective',-(m(end))^2);

prob.Constraints.cons=generate_constraints_RLV(time_breaks,point_each,r,s,V,gamma,m,theta,omega,T,beta);
prob.Constraints.cons1=r(1)==ri;
prob.Constraints.cons2=r(end)==1;
prob.Constraints.cons3=s(1)==0;
prob.Constraints.cons4=s(end)==1;
prob.Constraints.cons5=V(1)==1;
prob.Constraints.cons6=V(end)== 0.5/v0;
prob.Constraints.cons7=gamma(1)==gamma0;
prob.Constraints.cons8=gamma(end)==-90*pi/180;
prob.Constraints.cons9=m(1)==1;
prob.Constraints.cons10=beta<=10*pi/180;
prob.Constraints.cons11=beta>=-10*pi/180;
prob.Constraints.cons12=T<=T_max;
prob.Constraints.cons13=T>=0.5*T_max;
prob.Constraints.cons14=beta(end)==0;
prob.Constraints.cons15=theta(1)==theta0;
prob.Constraints.cons16=theta(end)==-pi/2;
prob.Constraints.cons17=omega(end)==0;
prob.Constraints.cons18=m>=21296.10751/m0;
prob.Constraints.cons19=generate_time_constraints(T_breaks,tau_f,point_each);
prob.Constraints.cons20=omega<=(20*pi/180);
prob.Constraints.cons21=omega>=-(20*pi/180);
prob.Constraints.cons22=generate_constraints_beta_der(time_breaks,point_each,beta,T);
prob.Constraints.cons23=omega(1)==omega0;
prob.Constraints.cons24=generate_segment_diff_at_ends(time_breaks,point_each,r,s,V,gamma,m,theta,omega,T,beta);

options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'Display','iter','ConstraintTolerance',1e-16,...
    'OptimalityTolerance',1e-16,'StepTolerance',1e-16,...
    'MaxFunctionEvaluations',5e3,'MaxIterations',750);

options = optimoptions(@fmincon,'Algorithm','interior-point','ConstraintTolerance',...
    1e-16,'OptimalityTolerance',1e-16,'StepTolerance',1e-16,'MaxFunctionEvaluations',750,'MaxIterations',750);

[sol,~,~,~] = solve(prob,init,'Options',options)
%,"ObjectiveDerivative",'auto',"ConstraintDerivative",'auto');
%[sol,fval,exitflag,output]=solve(prob,'Options',options);


tau_f=sol.tau_f;
Time_break=sol.T_breaks;
time_breaks_2=[0 Time_break];
time_breaks=[time_breaks_2,tau_f];
time_break=time_breaks;
rr=sol.r;
ss=sol.s;
VV=sol.V;
gammag=sol.gamma;
mm=sol.m;
thetat=sol.theta;
omegao=sol.omega;
TT=sol.T;
betab=sol.beta;
end