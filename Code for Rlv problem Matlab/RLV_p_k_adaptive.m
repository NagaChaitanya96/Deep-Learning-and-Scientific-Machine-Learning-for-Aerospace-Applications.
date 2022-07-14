% Solving the moon-lander problem to check step optimal control
% THE REASON GAMMA VALUE IS OF HIGHER ORDER IS DUE TO THE FINAL VELOCITY
% ->0. IF WE CHANGE THE FINAL VELOCITY TO 5.5M/S WHICH IS THE SAME INCASE
% OF THE SPACEX BOOSTER LANDING STAGE, WE ARE GETTING THE BEST POSSIBLE
% OUTCOME, WITH GOOD ACCURACY.

%% Defining global variables so that we dont have to change them in all the functions.
function [r_act,v_act,s_act,h_act,m_act,time_act,gamma_act,...
    theta_act,omega_act,Thrust_act,beta_act] = RLV_p_k_adaptive(h,v,s,hstep)
global R0 g0 m0 s0 v0 h0 omega0 states solg Cd S_ref Isp I_z l_com t_ref st break_time guess_matrix time_guess
global gamma0 omega0 theta0 controls point_each hstep
% clear h0 s0 v0
R0=6378*10^3; % Earth's radius
g0=9.81;
S_ref=10.75;
Cd=0.5;
Isp=300;
I_z=3346393;
l_com=9.778; % centre of mass distance
t_ref=32;

h0=h;        % No variation accepted %5000
s0=s;         % 850 and  855 are only working for this downrange distance %850
v0=v;         % acceptable range 325 -351 %325
% disp([h0 v0 s0])
gamma0 = -75*pi/180; %initial 75 changing gamma0 and theta0 in equal ranges gives good results
% whereas compared to individual variation in both the initial angles.
m0=26229.667;
theta0 = -75*pi/180; % initial 75
omega0 = 0.342;     % constraint -+ 20pi/180 = +- 0.3492,

%gamma theta omega initial conditions to be given
%% Transcribing OCP into NLP
N_max=10;
N_min=6;
% point_each=[17]; %best result with this;
% break_time=[];
final_time_guess=0.86;
count=0;
stop=1;
final_time_counter(1)=final_time_guess;
lll=1;
epsilon_array=[];
point_sum_array=[];
num_iter=1;

% while stop  %% 5 and epsilon 5e-14 works best
% point_each=[17 12]; %best result with this;
% break_time=[0.65];
% for ttt=1:2
err_f1 = 10;
tol = 1e-3;
ef = [];
while err_f1 >= tol
    % Nmax=12, Nmin=5, eps=1e-3, ttt=1:8 for good results
    if isempty(break_time)
        [time_breaks,r,s,V,gamma,m,theta,omega,T,beta]=NLP_solver_unity(point_each,final_time_guess);
        r1 =r;s1 =s;v1 =V;gamma1 = gamma;m1 = m; theta1 = theta;omega1 = omega;T1 = T; beta1 = beta;
    else % if there is any breaktime then this function will run and obtain the values after optimization.
        [time_breaks,r,s,V,gamma,m,theta,omega,T,beta]=NLP_solver_non_unity(point_each,break_time,guess_matrix,time_guess);
        r1 =r;s1 =s;v1 =V;gamma1 = gamma;m1 = m; theta1 = theta;omega1 = omega;T1 = T; beta1 = beta;
    end
    point_each_current=point_each   ;            % point_each = 17 initially now 13
    time_breaks_current=time_breaks  ;           % Time breaks = 0 : 0.86008 initial and final time
    
    
    % Why is the mean value equal to max(abs(r))?
    % r_mean=max(abs(r),[],'all');
    % s_mean=max(abs(s),[],'all');
    % V_mean=max(abs(V),[],'all');
    % gamma_mean=max(abs(gamma),[],'all');
    % m_mean=max(abs(m),[],'all');
    % theta_mean=max(abs(theta),[],'all');
    % omega_mean=max(abs(omega),[],'all');
    % Mean_array=[r_mean,s_mean,V_mean,gamma_mean,m_mean,theta_mean,omega_mean];
    %% Assessment of approx error term R and beta
    epsilon=1e-3;
    rho=2.8; % parameter for hp adaptiv. THIS PARAM IS USED TO CHECK
    % WHETHER TO DIVIDE THE NO OF COLLOC POINTS OR TO INCREASE THE DEGREE OF THE POLYNOMIAL
    T_b_new=[];
    M_new=[];
    states.r = [];
    states.s = [];
    states.V = [];
    states.m = [];
    states.gamma = [];
    states.theta = [];
    states.omega = [];
    states.time = [];
    for mm=1:length(point_each)
        [~,t_ref]=cheb(-1,1,point_each(mm)-1); % Chebyshev discretization of grid from [-1,1] in 17 points
        %% Problem occurs only when the number of points are less than the range given in interp1
        % Changing the time domain from 0 to tfinal using the below formula
        time_ch=0.5*(time_breaks(mm+1)-time_breaks(mm))*t_ref+0.5*(time_breaks(mm+1)+time_breaks(mm));
        if mm==1
            time_final=time_ch;
        else
            time_final=[time_final,time_ch(2:end)];
        end
        [~,indd]=ismembertol(time_ch,time_final);
        
        %    [LIA,LOCB] = ismembertol(A,B) also returns an array, LOCB, which
        %     contains an index location in B for each element in A which is
        %     a member of B.
        
        
        r_ind=r(indd);
        
        s_ind=s(indd);
        V_ind=V(indd);   %All are column vectors
        gamma_ind=gamma(indd);
        m_ind=m(indd);
        theta_ind=theta(indd);
        omega_ind=omega(indd);
        T_ind=T(indd);
        beta_ind=beta(indd);
        %% IN R -> removed residue of gamma
        
        [R,betta,time_mid]=generate_approx_error(time_ch,point_each(mm)...
            ,r_ind,s_ind,V_ind,gamma_ind,m_ind,theta_ind,omega_ind,T_ind,beta_ind);
        betta=betta.';
        epsilon_max=abs(max(R,[],'all'));
        epsilon_array=[epsilon_array epsilon_max];
        if epsilon_max<epsilon
            T_b_new=[T_b_new time_breaks(mm+1)];
            M_new=[M_new point_each(mm)];
            count=count+1;
            
        else
            if point_each(mm)<=N_max    % N_max is 10 since 17 > then this step is skipped.
                P_k=floor(max(log(epsilon_max/epsilon)/log(point_each(mm)),3)); %% Conditino given to increase the number of points inside the segments.
                % P_k=2;
                T_b_new=[T_b_new time_breaks(mm+1)];
                M_new=[M_new, point_each(mm)+P_k];
            else
                [T_knots,N_knots]=beta_break_algo(betta,time_mid,rho,N_min,N_max); %% FUNCTION
                %            disp('lala');
                T_b_new=[T_b_new T_knots time_breaks(mm+1)]; % breakout points
                M_new=[M_new N_knots];
            end
        end
    end
    if count==length(point_each) %&& num_iter>4
        stop=0;
    end
    %% Here the point Break happens
    point_each=M_new ;% here it changes the number of points
    break_time=T_b_new(1:end-1); % breaks at the initial time for the 1st part of t_break
    final_time_guess=T_b_new(end);
    
    
    point_sum_array=[point_sum_array length(break_time)];
    
    count=0;
    lll=lll+1;
    final_time_counter(lll)=T_b_new(end);
    %if abs(final_time_counter(lll)-final_time_counter(lll-1))<1e-13 && num_iter>4
    %   break
    %end
    
    %% Different between final time counter if < 1e-13 this breaks
    if abs(final_time_counter(lll)-final_time_counter(lll-1))<1e-13
        break
    end
    
    guess_matrix=[r'; s';V'; gamma'; m'; theta'; omega';T';beta'];
    time_guess=time_final;
    
    num_iter=num_iter+1;
    
    %% Final point residue
    rtf = states.r(end);
    Vtf = states.V(end);
    gammatf = states.gamma(end);
    mtf = states.m(end);
    omegatf = states.omega(end);
    stf = states.s(end);
    thetatf = states.theta(end);
    act_f = [r_ind V_ind gamma_ind m_ind omega_ind s_ind theta_ind];
    act_f = act_f(end,:)';
    pred_f  = [rtf;Vtf;gammatf;mtf;omegatf;stf;thetatf];
    % actual_final = [1.0;0.1/sqrt(g0*R0);...
    %     -pi/2;21296.10751/m0; 0;1.;-pi/2];
    actual_final = [1.0;5/v0;-pi/2;21296.10751/m0; 0;1.;-pi/2];
    error_at_final_states = actual_final - pred_f;
    % fprintf("error values of final states r,V,gamm,m,Omega,s,theta are: %d \n",error_at_final_states)
    err_f = abs(act_f - pred_f);
    % fprintf("error r :%d\n,error V :%d\n ,error gamma :%d\n ,error m :%d\n ,error Omega :%d\n ,error s :%d\n ,error theta :%d\n ",err_f)
    % fprintf("error r :%d\n,error V :%d\n ,error gamma :%d\n ,error m :%d\n ,error Omega :%d\n ,error s :%d\n ,error theta :%d\n ",error_at_final_states)

    fprintf("error r :%d\n,error V :%d\n ,error gamma :%d\n ,error m :%d\n ,error Omega :%d\n ,error s :%d\n ,error theta :%d\n ",err_f)
    err_f1 = max(err_f);
    
    ef = [ef,err_f1];
    sf = max(size(ef));
    disp([h0,v0,s0])
    i = 1;
    if err_f1 >= tol && sf > 4
        if rand > 0.5
            if rand > 0.5
                h0 = floor(h0);
                s0 = floor(s0);
                v0 = floor(v0)-0.25;
                ef =[];
                err_f1 = 10;
            else
                h0 = floor(h0);
                s0 = floor(s0);
                v0 = floor(v0)+1;
                ef = [];
                err_f1 = 10;
            end
        else
            if rand > 0.5
                h0 = floor(h0);
                s0 = floor(s0);
                v0 = ceil(v0)+0.25;
                ef =[];
                err_f1 = 10;
            else
                h0 = floor(h0);
                s0 = floor(s0);
                v0 = floor(v0)-1;
                ef = [];
                err_f1 = 10;
            end
        end
        disp([h0,v0,s0])
    end
    point_each;
    break_time;
end

%% Plots
point_each=point_each_current;
time_breaks=time_breaks_current;
cum_sum_jk=cumsum(point_each);

controls.time_interp=[];
controls.TT_interp=[];
controls.beta_interp=[];
omega_interp=[];


%% I needed to get the control values for all the corresponding states

% to get equal number of points distributed in the domain we need to do
% only 1 optimization problem, so that there will be no breaking points in
% between and our solution will be obtained from 0 to 0.8666 in steps size
% of 0.0001. If we are doing multiple optimizations, our optimizer will give us
% few breaking points and for each breaking point since we are dividing
% into 0.0001 step size, we will be getting total of different number of
% points for the system. So our idea of getting total number of control
% points for the given states might miss due to this infeasibility,
% however, we can devise a way that for each control point we can get the
% states by utilizing RK4 method , or verify if there is any already known
% values present in the system.

%h1 = 8.675016039500070e-05 h = 6.624053772881134e-05

time2=[];
for mm=1:length(point_each)
    [~,t_ref]=cheb(-1,1,point_each(mm)-1); % chebyshev time domain
    %t_ref=t_ref';
    % changing domain from [-1,1] to [0,t_end]
    time_ch=0.5*(time_breaks(mm+1)-time_breaks(mm))*t_ref+0.5*(time_breaks(mm+1)+time_breaks(mm));
    hstep1 = (time_ch(end)-time_ch(1))/hstep;
    tt=time_ch(1):hstep1:time_ch(end);
    time2 = [time2,tt];
    if mm==1
        
        TT=lagrange(tt,time_ch,T(1:cum_sum_jk(mm)).'); %breaks thrust into many number of lagrange points
        betab=lagrange(tt,time_ch,beta(1:cum_sum_jk(mm)).');
        omega0=lagrange(tt,time_ch,omega(1:cum_sum_jk(mm)).');
    else
        TT=lagrange(tt,time_ch,T(cum_sum_jk(mm-1)-mm+2:cum_sum_jk(mm)-mm+1).');
        betab=lagrange(tt,time_ch,beta(cum_sum_jk(mm-1)-mm+2:cum_sum_jk(mm)-mm+1).');
        omega0=lagrange(tt,time_ch,omega(cum_sum_jk(mm-1)-mm+2:cum_sum_jk(mm)-mm+1).');
    end
    controls.time_interp=[controls.time_interp tt];
    controls.TT_interp=[controls.TT_interp TT];
    controls.beta_interp=[controls.beta_interp betab];
    omega_interp=[omega_interp omega0];
end

%% Converting back to original states
Thrust_max = 756.222e3;
[r_act,v_act,s_act,h_act,m_act,time_act,gamma_act,...
    theta_act,omega_act,Thrust_act,beta_act] = convert_back_to_actual();
end
%%
% figure(1)
% plot(controls.time_interp,controls.TT_interp); hold on;
% plot(time_final.',T,'o');
%
% figure(2)
% plot(controls.time_interp,controls.beta_interp); hold on;
% plot(time_final.',beta,'o');
%
% figure(3)
% plot(controls.time_interp,omega_interp); hold on;
% plot(time_final.',omega,'o');
