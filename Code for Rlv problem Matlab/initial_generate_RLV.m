function [r_init,s_init,V_init,gamma_init,m_init,theta_init,...
    omega_init,T_init,beta_init,t_end]=initial_generate_RLV...
    (t_points,point_each,guess_matrix,time_guess)
% disp(point_each)
t_end=time_guess(end);
r=guess_matrix(1,:);
s=guess_matrix(2,:);
V=guess_matrix(3,:);
gamma=guess_matrix(4,:);
m=guess_matrix(5,:);
theta=guess_matrix(6,:);
omega=guess_matrix(7,:);
T=guess_matrix(8,:);
beta=guess_matrix(9,:);

Time=[t_points,t_end];
time_ref=[];
for i=1:length(point_each)
    [~,ref]=cheb(-1,1,point_each(i)-1);
    %[~,ref]=legendre(point_each(i)-1);
    temp=0.5*(Time(i+1)-Time(i))*ref+0.5*(Time(i+1)+Time(i));
    if i==1
        time_ref=temp;
    else
        time_ref=[time_ref temp(2:end)];
    end
end
% st = time_guess;
% if (st(end) >= time_guessed(end))
%     r_init=interp1(sol_guess.time,sol_guess.r,time_guessed);
%     % since the time guessed is >0.86 the rest all values ar;e taken as NAN
%     s_init=interp1(sol_guess.time,sol_guess.s,time_guessed);
%     V_init=interp1(sol_guess.time,sol_guess.V,time_guessed);
%     gamma_init=interp1(sol_guess.time,sol_guess.gamma,time_guessed);
%     m_init=interp1(sol_guess.time,sol_guess.m,time_guessed);
%     beta_init=interp1(sol_guess.time,sol_guess.beta,time_guessed);
%     T_init=interp1(sol_guess.time,sol_guess.T,time_guessed);
%     theta_init=interp1(sol_guess.time,sol_guess.theta,time_guessed);
%     omega_init=interp1(sol_guess.time,sol_guess.omega,time_guessed);
%
% elseif (st(end) <= time_guessed(end))
%
%
%     [r_init,s_init,V_init,gamma_init,m_init,theta_init,...
%         omega_init,T_init,beta_init,t_end_g]=initial_generate_RLV(...
%         0,point_each,guess_matrix,time_guess);
%     r_init';
%     t_end_g';
% end

% disp(time_ref)
% disp(time_guess)
% fprintf('\n the size of time_guess is %f x %f \n',size(time_guess))
% fprintf('the size of time_ref is %fx%f\n',size(time_ref))
% fprintf('the size of sol_guess.r is %fx%f\n',size(r))
% 
% if max(size(time_guess)) >= max(size(time_ref))
%     fprintf(" size @ left is > right")
% else 
%     fprintf("size @right is <left")
% end
if time_guess(end) < time_ref(end)
    point_each1 = [sum(point_each)-1];
    [r_init,s_init,V_init,gamma_init,m_init,theta_init,...
        omega_init,T_init,beta_init,t_end]=initial_generate_RLV(...
        0,point_each1,guess_matrix,time_guess);
else
    r_init=interp1(time_guess,r,time_ref);
    s_init=interp1(time_guess,s,time_ref);
    V_init=interp1(time_guess,V,time_ref);
    gamma_init=interp1(time_guess,gamma,time_ref);
    m_init=interp1(time_guess,m,time_ref);
    beta_init=interp1(time_guess,beta,time_ref);
    T_init=interp1(time_guess,T,time_ref);
    theta_init=interp1(time_guess,theta,time_ref);
    omega_init=interp1(time_guess,omega,time_ref);
end

end