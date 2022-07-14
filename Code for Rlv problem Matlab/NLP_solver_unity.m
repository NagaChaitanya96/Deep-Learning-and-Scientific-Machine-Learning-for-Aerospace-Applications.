% Function to solve for the cases where there is no break point i.e. during
% the initial part


% we have two ways of solving this problem
% 1. Either we create a new data file each time the time gets
% updated such that our sol_guess.time will be equal to the time_guessed.
% 2. We create a function to randomly generate interpollant values similar
% to the NLP_solver_non_unity has done.

function [time_break,rr,ss,VV,gammag,...
    mm,thetat,omegao,TT,betab]=NLP_solver_unity(point_each,fin_t_g)
global R0 g0 m0 s0 v0 h0 theta0 gamma0 omega0 solg st break_time %guess_matrix time_guess
global init prob
N=point_each-1;  % 47 s=950 for superfast
%state variables
r=optimvar('r',N+1,1); % creates a N+1 -by-1 vector of r values
s=optimvar('s',N+1,1);
V=optimvar('V',N+1,1);
gamma=optimvar('gamma',N+1,1);
m=optimvar('m',N+1,1);
theta=optimvar('theta',N+1,1);
omega=optimvar('omega',N+1,1);
tau_f=optimvar('tau_f',1,1);

% Control variables
T=optimvar('T',N+1,1);
beta=optimvar('beta',N+1,1);

S_ref=10.75;
ri=(R0+h0)/R0;
%rf=1 as the rocket will have landed.

I_z=3346393;
l_com=9.778; % centre of mass distance
T_max=756.222e3/(m0*g0);
t_ref=32;
%% BECAUSE OF THIS VARIATION IN TIME WE GET ERROR.

[D,ref]=cheb(-1,1,N); % D,ref = derivative, reference time points N+1 vector
%% All the values in the below file are normalized and so we have range very low.
sol_guess=matfile('inicon_27.mat'); %1*28 matrix
solg = [sol_guess.beta' sol_guess.gamma' sol_guess.m' sol_guess.omega' sol_guess.r' sol_guess.s' sol_guess.T' sol_guess.V' sol_guess.theta'];
time_guessed=0.5*(fin_t_g-0)*ref'+0.5*(fin_t_g+0);% taking N+1 points of time guessed
%% This will give the 16*1 vector for each of the states.
% sol_guess.time =time_guessed
% sol_guess.time(end) = time_guessed(end) then only we are getting the
% required values for our optimization problem
st = sol_guess.time;

% fprintf('\n the size of sol_guess.time is %fx%f\n',size(sol_guess.time))
% fprintf('the size of time_guessed is %fx%f\n',size(time_guessed))
% fprintf('the size of sol_guess.r is %fx%f\n',size(sol_guess.r))

% if max(size(sol_guess.time)) >= max(size(time_guessed))
%     fprintf(" size @ left is > right")
% else 
%     fprintf("size @right is <left")
% end

if (st(end) >= time_guessed(end))
    r_init=interp1(sol_guess.time,sol_guess.r,time_guessed);
    % since the time guessed is >0.86 the rest all values ar;e taken as NAN
    s_init=interp1(sol_guess.time,sol_guess.s,time_guessed);
    V_init=interp1(sol_guess.time,sol_guess.V,time_guessed);
    gamma_init=interp1(sol_guess.time,sol_guess.gamma,time_guessed);
    m_init=interp1(sol_guess.time,sol_guess.m,time_guessed);
    beta_init=interp1(sol_guess.time,sol_guess.beta,time_guessed);
    T_init=interp1(sol_guess.time,sol_guess.T,time_guessed);
    theta_init=interp1(sol_guess.time,sol_guess.theta,time_guessed);
    omega_init=interp1(sol_guess.time,sol_guess.omega,time_guessed);
    
elseif (st(end) <= time_guessed(end))
    [r_init,s_init,V_init,gamma_init,m_init,theta_init,...
        omega_init,T_init,beta_init,t_end_g]=initial_generate_RLV(...
        0,point_each,guess_matrix,time_guess);
    r_init';
    t_end_g';
end




% r_init=(1-ri)/2*ref+(1+ri)/2;
% s_init=1/R0*((s0-0)/2*ref+(s0+0)/2);
% V_init=1/sqrt(R0*g0)*((1e-2-v0)/2*ref+(1e-2+v0)/2);
% gamma_init=(-90*pi/180+65*pi/180)/2*ref+(-90*pi/180-65*pi/180)/2;
% m_init=1/m0*((21296.10751-m0)/2*ref+(21296.10751+m0)/2);
% beta_init=(0-0.173)/2*ref+(0.173+0)/2;
% theta_init=gamma_init;
% omega_init=1/sqrt(g0/R0)*(0.5*(0+0.08356)*ref+0.5*(0-0.08356));
%
% init.r=r_init.';
% init.s=s_init.';
% init.V=V_init.';
% init.gamma=gamma_init.';
% init.m=m_init.';
% init.tau_f=32/sqrt(R0/g0);
% init.theta=theta_init.';
% %init.omega=omega_init.';
% init.omega=zeros(N+1,1);
%
% init.T=0.5*T_max*ones(N+1,1);
% %init.alpha=zeros(N+1,1);
% %init.beta=beta_init;
% init.beta=zeros(N+1,1);

%% INitial values after interpolation; supplied which is the same dimension of time domain discretized
if max(size(T)) ~= max(size(T_init))
    no = abs(max(size(T)) - max(size(T_init)));
    init.tau_f=fin_t_g;
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
    init.tau_f=fin_t_g;
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


% init.r=r_init';
% init.s=s_init';
% init.V=V_init';
% init.gamma=gamma_init';
% init.m=m_init';
% init.theta=theta_init';
% init.omega=omega_init';
% init.T=T_init';
% init.beta=beta_init';
% init.tau_f=fin_t_g;

prob=optimproblem('Objective',-(m(end))^2);     % objective function is minimum fuel consumed.
Isp=300;

%% lift is assumed constant since very low impact.
%Cl=2.3*alpha;
%Cd=0.0975+0.1819*Cl.^2;

Cd=0.5;
%L=0.5*1.225*((V*sqrt(g0*R0)).^2).*Cl*S_ref/(m0*g0);
Dr=0.5*1.225*((V*v0).^2).*Cd*S_ref/(m0*g0);
prob.Constraints.cons=(2/(tau_f))*D*r==(v0*t_ref/R0)*V.*sin(gamma);
prob.Constraints.cons1=(2/(tau_f))*D*s==(v0*t_ref/s0)*V.*cos(gamma);
prob.Constraints.cons2=(2/(tau_f))*D*V==(g0*t_ref/v0)*((-T.*cos(beta-gamma+theta)-Dr)./m-sin(gamma)./(r.^2));
prob.Constraints.cons3=(2/(tau_f))*D*gamma==(g0*t_ref/v0)*((-T.*sin(beta-gamma+theta))./(m.*V)-cos(gamma)./((r.^2).*V));
prob.Constraints.cons4=(2/(tau_f))*D*m==-T/Isp*t_ref;
prob.Constraints.cons5=(2/(tau_f))*D*theta==omega*t_ref;
prob.Constraints.cons6=(2/(tau_f))*D*omega==-T.*sin(beta)*l_com/I_z*m0*g0*t_ref;
prob.Constraints.cons7=r(1)==ri;
prob.Constraints.cons8=r(N+1)==1;
prob.Constraints.cons9=s(1)==0;
prob.Constraints.cons10=s(N+1)==1;
prob.Constraints.cons11=V(1)==1;
prob.Constraints.cons12=V(N+1)== 0.5/v0;
prob.Constraints.cons13=gamma(1)== gamma0;
prob.Constraints.cons14=gamma(N+1)==-90*pi/180;
prob.Constraints.cons15=m(1)==1;
%prob.Constraints.cons14=alpha<=15*pi/180;
%prob.Constraints.cons15=alpha>=-15*pi/180;
prob.Constraints.cons16=beta<=10*pi/180;
prob.Constraints.cons17=beta>=-10*pi/180;
prob.Constraints.cons18=T<=T_max;
prob.Constraints.cons19=T>=0.5*T_max;
prob.Constraints.cons20=beta(N+1)==0;
prob.Constraints.cons21=theta(1)== theta0;
prob.Constraints.cons22=theta(N+1)==-pi/2;
prob.Constraints.cons23=omega(N+1)==0;
prob.Constraints.cons24=m>=21296.10751/m0;
prob.Constraints.cons25=omega<=(20*pi/180);
prob.Constraints.cons26=omega>=-(20*pi/180);
prob.Constraints.cons27=omega(1)== omega0;
prob.Constraints.cons28=D*beta<= 10*tau_f/2*t_ref*pi/180;
prob.Constraints.cons29=D*beta>=-10*tau_f/2*t_ref*pi/180;
prob.Constraints.cons30=D*T<=tau_f/2*25;
prob.Constraints.cons31=D*T>=-tau_f/2*25;

%% Need to impose few more constraints, that r>=0; v>=0; s>=0; However from the plots , we can observe the
% anomaly is really less, we can neglect these constraints.
% disp(init)
% %[sol,fval,exitflag,output] = solve(prob,init);
%% SOLVING THE PROBLEM USING FMINCON
% options = optimoptions(@fmincon,'Algorithm','interior-point','Display','iter','ConstraintTolerance',1e-16,'OptimalityTolerance',1e-16,'StepTolerance',1e-16,'MaxFunctionEvaluations',1e4,'MaxIterations',1e4);

options = optimoptions(@fmincon,'Algorithm','interior-point','ConstraintTolerance',1e-16,...
    'OptimalityTolerance',1e-16,'StepTolerance',1e-16,'MaxFunctionEvaluations',2500,'MaxIterations',2500);

[sol,~,~,~] = solve(prob,init,'Options',options);
tau_f=sol.tau_f;
% time_breaks=[0,tau_f/2,tau_f;
time_breaks=[0,tau_f/3,2*tau_f/3 ,tau_f];

%%%
time_break=time_breaks
%%%
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