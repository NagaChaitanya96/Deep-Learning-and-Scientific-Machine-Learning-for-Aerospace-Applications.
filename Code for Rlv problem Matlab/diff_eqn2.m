function dy=diff_eqn2(t,y,T_i,beta_i,time)
global R0 g0 m0 s0 v0 h0   omega0 states solg Cd S_ref Isp I_z l_com t_ref
%y=[r,s,V,gamma,m,theta,omega]
y=y';
Cd=0.5;
S_ref=10.75;
Isp=300;
%% After Debugging 3 -> 4 
I_z=3346393;
l_com=9.778; % centre of mass distance
t_ref=32;
Dr=0.5*1.225*((y(4)*v0).^2).*Cd*S_ref/(m0*g0);
%% Langrage interpolation gives the interpolation values in between the domain in steps
% a = 0:10
% b = a.^2
% aa = linspace(0,10) %divides the space into 100 points equi distance
% bb = lagrange(aa,a,b) % interpolates the polynomial within the 100
% inbetween points.
% plot(a,b,'o',aa,bb,'.')

T=lagrange(t,time,T_i); % lagrange(time_interp, time_ch, T) size = size(time_interp)
beta=lagrange(t,time,beta_i);
%T=interp1(time,T_i,t,'pchip');
%beta=interp1(time,beta_i,t);

%%  why did he multiply with (v0*t_ref/R0), g0*t_ref/R0
% it is reasonable to multiply with m0*g0*t_ref since I_Z and Isp are non
% dimensionalized.

dy(1,1)=y(3)*sin(y(4))*(v0*t_ref/R0); % dy' = V'sin(gamma). WHy is there an extra *(v0*t_ref/R0);
dy(2,1)=y(3)*cos(y(4))*(v0*t_ref/s0);
dy(3,1)=((-T*cos(beta-y(4)+y(6))-Dr)/y(5)-sin(y(4))/(y(1)^2))*(g0*t_ref/v0);
dy(4,1)=((-T*sin(beta-y(4)+y(6)))/(y(5)*y(3))-cos(y(4))./((y(1)^2)*y(3)))*(g0*t_ref/v0);
dy(5,1)=-T/Isp*t_ref;
dy(6,1)=y(7)*t_ref;     % order of omega is 100 > rest of states since its multipled by t_ref
dy(7,1)=-T*sin(beta)*l_com/I_z*(m0*g0*t_ref);



end