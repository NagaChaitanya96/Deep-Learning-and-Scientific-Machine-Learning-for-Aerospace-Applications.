function [R,beta,time_mid]=generate_approx_error(timef,point_each,r,s,V,gamma,m,theta,omega,T,beta)
% r_indd,....valeus are the input here as per the main program

time=timef;

for i=1:point_each-1
    time_mid(i)=0.5*(time(i)+time(i+1));
end
% time_mid1  = (time_ch(1:16)+time_ch(2:17))*0.5

%time_mid=time;

time=time';                                 % size(time_ch)*1  0
time_mid=time_mid';                         % (size(time_ch) - 1)*1  
rr_mid=lagrange(time_mid,time,r);
%rr_mid_der=lagrange1(time_mid,time,r,1);
ss_mid=lagrange(time_mid,time,s);
%ss_mid_der=lagrange1(time_mid,time,s,1);
VV_mid=lagrange(time_mid,time,V);
%VV_mid_der=lagrange1(time_mid,time,V,1);
gamma_mid=lagrange(time_mid,time,gamma);
%gamma_mid_der=lagrange1(time_mid,time,gamma,1);
m_mid=lagrange(time_mid,time,m);
%m_mid_der=lagrange1(time_mid,time,m,1);
theta_mid=lagrange(time_mid,time,theta);
%theta_mid_der=lagrange1(time_mid,time,theta,1);
omega_mid=lagrange(time_mid,time,omega);
%omega_mid_der=lagrange1(time_mid,time,omega,1);
T_mid=lagrange(time_mid,time,T);
%T_mid_der=lagrange1(time_mid,time,T,1);
beta_mid=lagrange(time_mid,time,beta);
%beta_mid_der=lagrange1(time_mid,time,beta,1);
%% if N = 17 ; N_mid = 16*1 vector
% disp([rr_mid,ss_mid,VV_mid,gamma_mid,m_mid,theta_mid,omega_mid,T_mid,beta_mid])

% r_indd, s_indd ... will be sent here into the rk4_integral, and we obtain
% the vector of values which are interpolated at in stepsize of 1e-4
% time_mid1  = (time(1:16)+time(2:17))*0.5;
% rr_mid1 = lagrange(time_mid1,time,r);
% rr_mid == rr_mid1
% The below integral gives the state values at the mid point values at the interpolated points.

[r_int,s_int,V_int,gamma_int,m_int,theta_int,omega_int]=rk4_integral(timef,...
point_each,r,s,V,gamma,m,theta,omega,T,beta);

Cd=0.5;
S_ref=10.75;
I_z=3346393;
l_com=9.778; % centre of mass distance
Isp=300;

t_ref=32;

X=[rr_mid, ss_mid, VV_mid, gamma_mid, m_mid, theta_mid, omega_mid];
%disp(size(X));
X_int=[r_int, s_int, V_int, gamma_int, m_int, theta_int, omega_int];
%disp(size(F));
%disp(size(D_mid));
R=abs(X-X_int);
% disp(size(R));
%% Why did he consider Mean_array = abs(Max(rr_mid))? that will give maximum instead of the mean values.
Mean_array=abs([max(rr_mid), max(ss_mid), max(VV_mid), max(gamma_mid), max(m_mid), max(theta_mid), max(omega_mid)]);

MA=ones(point_each-1,1)*Mean_array;
%R=R./MA;
R(:,4)=[]; %% Gamma mid - gamma-initial is removed here? Why?
ind=max(R);
ind1=ind==max(ind);
r1=R(:,ind1);
r_avg=mean(r1);
beta=r1/r_avg; % This beta is the pg no 86 in book
%R(:,4)=[];
[~,ref]=cheb(-1,1,point_each-1);
%[~,ref]=legendre(point_each-1);
time=0.5*(timef(end)-timef(1))*ref+0.5*(timef(end)+timef(1));
for i=1:point_each-1
    time_mid(i)=0.5*(time(i)+time(i+1));
end

end